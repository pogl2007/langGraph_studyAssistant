"""API: загрузка и её безопасность, статус индексации, SSE-чат, тест через resume, обратная связь."""

import json
import time
from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from app.api.main import Services, create_app
from app.graph.builder import build_graph
from app.graph.nodes import AgentNodes
from app.ingest.indexer import Indexer

from .conftest import ScriptedLLM, make_pdf, make_settings
from .test_graph import LECTURE, FakeImages, quiz_json


@pytest.fixture
def api(embedder, store, sparse, retriever, tmp_path):
    settings = make_settings(data_dir=tmp_path, max_upload_mb=1, public_api_url="http://api.test")
    llm = ScriptedLLM(answer="Поток равен заряду внутри поверхности [1].", quiz=quiz_json(2))
    indexer = Indexer(embedder, store, 40, 5, sparse=sparse, chunk_min_tokens=3, files_dir=settings.files_dir)
    graph = build_graph(
        AgentNodes(retriever, llm, settings, list_sections=store.list_sections, images=FakeImages()),
        checkpointer=InMemorySaver(),
        store=InMemoryStore(),
    )

    @asynccontextmanager
    async def factory():
        yield Services(graph=graph, indexer=indexer, store=store, settings=settings)

    with TestClient(create_app(factory)) as client:
        client.settings = settings
        yield client


def sse(response) -> list[tuple[str, dict]]:
    events, name = [], None
    for line in response.iter_lines():
        if line.startswith("event:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            events.append((name, json.loads(line.split(":", 1)[1])))
    return events


def upload(api, name="lecture.pdf", data=None, course="physics"):
    return api.post("/documents", files={"file": (name, data or make_pdf(LECTURE), "application/pdf")},
                    data={"course": course})


def wait_task(api, task_id: str) -> dict:
    for _ in range(100):
        status = api.get(f"/documents/tasks/{task_id}").json()
        if status["status"] in ("done", "failed"):
            return status
        time.sleep(0.05)
    raise AssertionError("индексация не завершилась")


def test_upload_returns_202_and_task_finishes(api):
    r = upload(api)
    assert r.status_code == 202
    assert r.headers["x-request-id"]
    status = wait_task(api, r.json()["task_id"])
    assert status["status"] == "done"
    assert status["result"]["chunks"] == 3

    [doc] = api.get("/documents", params={"course": "physics"}).json()
    assert doc["filename"] == "lecture.pdf"
    pdf = api.get(f"/documents/{doc['file_id']}/file")
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF-")

    assert api.delete(f"/documents/{doc['file_id']}").status_code == 204
    assert api.get("/documents").json() == []


@pytest.mark.parametrize(
    ("name", "data", "status"),
    [
        ("notes.exe", b"MZ...", 415),  # расширение
        ("fake.pdf", b"<html>not a pdf</html>", 415),  # содержимое не PDF
        ("big.pdf", b"%PDF-" + b"0" * (1024 * 1024 + 10), 413),  # размер
        ("../..", b"%PDF-1.7", 415),  # нет расширения после очистки пути
    ],
    ids=["extension", "not-pdf", "too-big", "no-name"],
)
def test_upload_validation(api, name, data, status):
    assert upload(api, name=name, data=data).status_code == status


def test_path_traversal_in_filename_is_neutralized(api):
    r = upload(api, name="../../etc/lecture.pdf")
    assert r.status_code == 202
    assert r.json()["filename"] == "lecture.pdf"
    wait_task(api, r.json()["task_id"])
    files = list(api.settings.files_dir.iterdir())
    assert all(f.parent == api.settings.files_dir for f in files)


def test_file_endpoint_rejects_non_uuid(api):
    assert api.get("/documents/..%2F..%2Fsecrets/file").status_code == 404
    assert api.delete("/documents/not-a-uuid").status_code == 404


def test_unknown_task_is_404(api):
    assert api.get("/documents/tasks/nope").status_code == 404


def test_chat_streams_nodes_tokens_and_citations(api):
    wait_task(api, upload(api).json()["task_id"])
    with api.stream("POST", "/chat", json={"message": "Что говорит теорема Гаусса о потоке напряженности?",
                                           "user_id": "u1", "course": "physics"}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        events = sse(r)

    names = [n for n, _ in events]
    assert names[0] == "start" and names[-1] == "done"
    assert [d["node"] for n, d in events if n == "node"] == ["route", "retrieve", "grade", "generate"]
    assert "token" in names and "status" in names
    answer = next(d for n, d in events if n == "answer")
    assert answer["text"].endswith("[lecture.pdf, стр. 2].")
    assert answer["citations"][0]["url"].startswith("http://api.test/documents/")
    assert answer["citations"][0]["url"].endswith("#page=2")


def test_quiz_over_sse_then_resume(api):
    wait_task(api, upload(api).json()["task_id"])
    with api.stream("POST", "/chat", json={"message": "Составь тест по теме Закон Кулона", "thread_id": "q1",
                                           "user_id": "u1", "course": "physics"}) as r:
        events = sse(r)
    quiz = next(d for n, d in events if n == "quiz")
    assert len(quiz["questions"]) == 2
    assert "correct" not in json.dumps(quiz)
    assert events[-1] == ("done", {"thread_id": "q1", "interrupted": True})

    with api.stream("POST", "/chat/resume", json={"thread_id": "q1", "user_id": "u1", "answers": [0, 0]}) as r:
        events = sse(r)
    answer = next(d for n, d in events if n == "answer")
    assert answer["text"].startswith("Результат:")


def test_resume_without_pending_quiz_is_an_error_event(api):
    with api.stream("POST", "/chat/resume", json={"thread_id": "empty", "answers": [0]}) as r:
        events = sse(r)
    assert events[0][0] == "error"


def test_feedback_is_appended(api):
    body = {"thread_id": "t1", "rating": -1, "question": "q", "answer": "a", "comment": "неточно"}
    assert api.post("/feedback", json=body).status_code == 204
    assert api.post("/feedback", json={**body, "rating": 1}).status_code == 204
    lines = api.settings.feedback_path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["rating"] for line in lines] == [-1, 1]
    assert api.post("/feedback", json={**body, "rating": 5}).status_code == 422


def test_health(api):
    assert api.get("/health").json()["status"] == "ok"


def test_draw_returns_image_url_that_serves_jpeg(api):
    wait_task(api, upload(api).json()["task_id"])
    with api.stream("POST", "/chat", json={"message": "Нарисуй схему теорема Гаусса", "course": "physics"}) as r:
        events = sse(r)
    answer = next(d for n, d in events if n == "answer")
    [img] = answer["images"]
    assert img["url"] == f"http://api.test/images/{img['image_id']}.jpg"

    got = api.get(f"/images/{img['image_id']}.jpg")
    assert got.status_code == 200
    assert got.headers["content-type"] == "image/jpeg"
    assert got.content == b"\xff\xd8fake-jpeg"


def test_image_endpoint_rejects_non_hex_ids(api):
    assert api.get("/images/..%2F..%2Fsecrets.jpg").status_code == 404
    assert api.get("/images/" + "0" * 32 + ".jpg").status_code == 404  # корректный id, но файла нет
