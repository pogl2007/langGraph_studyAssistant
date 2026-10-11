"""HTTP API ассистента.

    POST   /documents                 загрузка PDF → 202 и task_id, индексация в фоне
    GET    /documents/tasks/{task_id} статус индексации
    GET    /documents                 список файлов
    GET    /documents/{file_id}/file  сам PDF — ссылки цитат открывают его на нужной странице
    DELETE /documents/{file_id}
    POST   /chat                      ответ агента стримом SSE: узлы графа, токены, цитаты, тест
    POST   /chat/resume               ответы на тест (продолжение графа после interrupt)
    POST   /feedback                  👍 / 👎 на ответ
    GET    /health

Запуск: uvicorn app.api.main:app --port 8000
"""

import asyncio
import json
import logging
import re
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.sse import EventSourceResponse, ServerSentEvent
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from app.api.observability import RequestIdMiddleware, request_id, setup_logging
from app.api.tasks import IndexTasks
from app.config import Settings, get_settings
from app.graph.state import AgentContext
from app.ingest.indexer import Indexer, safe_filename
from app.retrieval.store import ChunkStore

log = logging.getLogger(__name__)

_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_IMAGE_ID = re.compile(r"^[0-9a-f]{32}$")


@dataclass
class Services:
    graph: Any
    indexer: Indexer
    store: ChunkStore
    settings: Settings


@asynccontextmanager
async def production_services() -> AsyncIterator[Services]:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from langgraph.store.sqlite.aio import AsyncSqliteStore

    from app.llm.client import LLMClient
    from app.services import build_agent, build_images, get_indexer, get_store, warm_up

    s = get_settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    log.info("загружаю модели")
    await asyncio.to_thread(warm_up)  # эмбеддинги, BM25, реранкер — до первого запроса, а не во время
    llm = LLMClient.from_settings(s)
    async with (
        AsyncSqliteSaver.from_conn_string(str(s.checkpoints_path)) as checkpointer,
        AsyncSqliteStore.from_conn_string(str(s.memory_path)) as memory_store,
    ):
        await checkpointer.setup()
        await memory_store.setup()
        images = build_images(s)
        graph = build_agent(llm, checkpointer=checkpointer, store=memory_store, settings=s, images=images)
        log.info("готов: модель %s, коллекция %s", llm.model, s.qdrant_collection)
        try:
            yield Services(graph=graph, indexer=get_indexer(), store=get_store(), settings=s)
        finally:
            await llm.aclose()
            if images is not None:
                await images.aclose()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    thread_id: str | None = Field(default=None, max_length=100)
    user_id: str = Field(default="anonymous", max_length=100)
    course: str | None = Field(default=None, max_length=100)


class ResumeRequest(BaseModel):
    thread_id: str = Field(max_length=100)
    user_id: str = Field(default="anonymous", max_length=100)
    course: str | None = Field(default=None, max_length=100)
    answers: list[int | None]


class FeedbackRequest(BaseModel):
    thread_id: str = Field(max_length=100)
    rating: Literal[1, -1]
    question: str = Field(default="", max_length=4000)
    answer: str = Field(default="", max_length=20000)
    comment: str = Field(default="", max_length=2000)


def create_app(services_factory: Callable[[], AbstractAsyncContextManager[Services]] = production_services) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging()
        async with services_factory() as services:
            app.state.services = services
            app.state.tasks = IndexTasks(services.indexer)
            yield
            await app.state.tasks.shutdown()

    app = FastAPI(title="Study Assistant", version="0.2.0", lifespan=lifespan)
    app.add_middleware(RequestIdMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in get_settings().cors_origins.split(",") if o.strip()],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def svc(request: Request) -> Services:
        return request.app.state.services

    @app.get("/health")
    async def health(request: Request):
        s = svc(request)
        try:
            await asyncio.to_thread(s.store.client.get_collections)
        except Exception as e:
            raise HTTPException(503, f"Qdrant недоступен: {e}") from e
        return {"status": "ok", "collection": s.settings.qdrant_collection, "indexed": await asyncio.to_thread(s.store.exists)}

    # ---------- документы ----------

    @app.post("/documents", status_code=202)
    async def upload(request: Request, file: UploadFile = File(...), course: str = Form("default")):
        s = svc(request)
        name = file.filename or ""
        if not name.lower().endswith(".pdf"):
            raise HTTPException(415, "Пока поддерживается только PDF")
        try:
            filename = safe_filename(name)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        if not re.fullmatch(r"[\w .\-]{1,100}", course):
            raise HTTPException(400, "Недопустимое название курса")
        limit = s.settings.max_upload_mb * 1024 * 1024
        data = await file.read(limit + 1)  # читаем на байт больше лимита — так видно превышение
        if len(data) > limit:
            raise HTTPException(413, f"Файл больше {s.settings.max_upload_mb} МБ")
        if not data.startswith(b"%PDF-"):
            raise HTTPException(415, "Файл не похож на PDF")
        task = request.app.state.tasks.submit(filename, data, course)
        return {"task_id": task.task_id, "status": task.status, "filename": filename}

    @app.get("/documents/tasks/{task_id}")
    async def task_status(request: Request, task_id: str):
        task = request.app.state.tasks.get(task_id)
        if task is None:
            raise HTTPException(404, "Задача не найдена")
        return task.public()

    @app.get("/documents")
    async def list_documents(request: Request, course: str | None = None):
        return await asyncio.to_thread(svc(request).store.list_files, course)

    @app.get("/images/{image_id}.jpg")
    async def image(request: Request, image_id: str):
        path = svc(request).settings.images_dir / f"{image_id}.jpg"
        if not _IMAGE_ID.match(image_id) or not path.exists():  # id — только hex: путь не собрать из ввода
            raise HTTPException(404, "Картинка не найдена")
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/documents/{file_id}/file")
    async def document_file(request: Request, file_id: str):
        s = svc(request)
        path = s.settings.files_dir / f"{file_id}.pdf"
        if not _UUID.match(file_id) or not path.exists():  # file_id — только UUID: путь не собрать из ввода
            raise HTTPException(404, "Файл не найден")
        meta = await asyncio.to_thread(s.store.get_file, file_id)
        filename = meta["filename"] if meta else f"{file_id}.pdf"
        return FileResponse(path, media_type="application/pdf", filename=filename, content_disposition_type="inline")

    @app.delete("/documents/{file_id}", status_code=204)
    async def delete_document(request: Request, file_id: str):
        s = svc(request)
        if not _UUID.match(file_id) or await asyncio.to_thread(s.store.get_file, file_id) is None:
            raise HTTPException(404, "Файл не найден")
        await asyncio.to_thread(s.indexer.delete, file_id)
        return Response(status_code=204)

    # ---------- чат ----------

    @app.post("/chat", response_class=EventSourceResponse)
    async def chat(request: Request, body: ChatRequest) -> AsyncIterator[ServerSentEvent]:
        thread_id = body.thread_id or uuid.uuid4().hex
        payload = {"messages": [HumanMessage(body.message)]}
        async for event in _run_graph(svc(request), payload, thread_id, body.user_id, body.course):
            yield event

    @app.post("/chat/resume", response_class=EventSourceResponse)
    async def resume(request: Request, body: ResumeRequest) -> AsyncIterator[ServerSentEvent]:
        s = svc(request)
        state = await s.graph.aget_state({"configurable": {"thread_id": body.thread_id}})
        if not any(t.interrupts for t in state.tasks):
            yield ServerSentEvent(event="error", data={"message": "В этом диалоге нет теста, ожидающего ответов"})
            return
        async for event in _run_graph(s, Command(resume=body.answers), body.thread_id, body.user_id, body.course):
            yield event

    @app.post("/feedback", status_code=204)
    async def feedback(request: Request, body: FeedbackRequest):
        s = svc(request)
        record = {**body.model_dump(), "ts": datetime.now(UTC).isoformat(timespec="seconds"), "request_id": request_id.get()}
        line = json.dumps(record, ensure_ascii=False) + "\n"
        await asyncio.to_thread(_append, s.settings.feedback_path, line)
        return Response(status_code=204)

    return app


async def _run_graph(s: Services, payload, thread_id: str, user_id: str, course: str | None):
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": s.settings.recursion_limit}
    context = AgentContext(user_id=user_id, course=course)
    yield ServerSentEvent(event="start", data={"thread_id": thread_id})
    interrupted = False
    try:
        async for mode, chunk in s.graph.astream(payload, config, context=context, stream_mode=["updates", "custom"]):
            if mode == "custom":
                yield ServerSentEvent(event=chunk.get("type", "custom"), data=chunk)
                continue
            for node, update in chunk.items():
                if node == "__interrupt__":
                    interrupted = True
                    yield ServerSentEvent(event="quiz", data=update[0].value)
                else:
                    yield ServerSentEvent(event="node", data={"node": node})
        if not interrupted:
            values = (await s.graph.aget_state(config)).values
            yield ServerSentEvent(event="answer", data={
                "text": values.get("answer", ""),
                "citations": [_with_url(c, s.settings) for c in values.get("citations", [])],
                "no_answer": values.get("no_answer", False),
                "route": values.get("route"),
                "images": [
                    {**img, "url": f"{s.settings.public_api_url}/images/{img['image_id']}.jpg"}
                    for img in values.get("images", [])
                ],
            })
    except Exception as e:
        log.exception("граф упал")
        yield ServerSentEvent(event="error", data={"message": f"Ошибка агента: {type(e).__name__}"})
    yield ServerSentEvent(event="done", data={"thread_id": thread_id, "interrupted": interrupted})


def _with_url(citation: dict, s: Settings) -> dict:
    if citation.get("file_id"):
        citation = {**citation, "url": f"{s.public_api_url}/documents/{citation['file_id']}/file#page={citation['page']}"}
    return citation


def _append(path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line)


app = create_app()
