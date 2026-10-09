"""Фоновая индексация: загрузка отвечает 202 сразу, а парсинг и эмбеддинги идут отдельной задачей.

Индексация — чистый CPU: в event loop она бы заблокировала все запросы, включая /health,
поэтому выполняется в потоке через asyncio.to_thread. Одновременно — одна задача: параллельные
индексации всё равно делят одни и те же ядра и только мешают друг другу.

Реестр задач в памяти процесса: после перезапуска статусы теряются. Для нескольких воркеров
нужна внешняя очередь (Redis + arq/Celery) — см. DECISIONS.md.
"""

import asyncio
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.ingest.indexer import Indexer

log = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class IndexTask:
    task_id: str
    filename: str
    course: str
    status: str = "queued"  # queued | running | done | failed
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: str = field(default_factory=_now)
    finished_at: str | None = None

    def public(self) -> dict[str, Any]:
        return asdict(self)


class IndexTasks:
    def __init__(self, indexer: Indexer, concurrency: int = 1):
        self.indexer = indexer
        self._tasks: dict[str, IndexTask] = {}
        self._running: set[asyncio.Task] = set()
        self._sem = asyncio.Semaphore(concurrency)

    def submit(self, filename: str, data: bytes, course: str) -> IndexTask:
        task = IndexTask(task_id=uuid.uuid4().hex, filename=filename, course=course)
        self._tasks[task.task_id] = task
        job = asyncio.create_task(self._run(task, data))
        self._running.add(job)  # без ссылки asyncio может собрать задачу сборщиком мусора
        job.add_done_callback(self._running.discard)
        return task

    def get(self, task_id: str) -> IndexTask | None:
        return self._tasks.get(task_id)

    async def _run(self, task: IndexTask, data: bytes) -> None:
        async with self._sem:
            task.status = "running"
            try:
                result = await asyncio.to_thread(self.indexer.index_pdf, task.filename, data, task.course)
                task.result = asdict(result)
                task.status = "done"
            except Exception as e:
                log.exception("индексация %s упала", task.filename)
                task.error = str(e)[:500]
                task.status = "failed"
            finally:
                task.finished_at = _now()

    async def wait_all(self) -> None:
        if self._running:
            await asyncio.gather(*self._running, return_exceptions=True)

    async def shutdown(self) -> None:
        for job in list(self._running):
            job.cancel()
        await self.wait_all()
