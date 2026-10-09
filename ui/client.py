"""HTTP-клиент интерфейса к API: обычные запросы и разбор потока SSE."""

import json
from collections.abc import Iterator
from typing import Any

import httpx


class ApiClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.http = httpx.Client(base_url=self.base_url, timeout=httpx.Timeout(30, connect=5))

    def health(self) -> dict[str, Any]:
        return self.http.get("/health").raise_for_status().json()

    def upload(self, filename: str, data: bytes, course: str) -> dict[str, Any]:
        r = self.http.post("/documents", files={"file": (filename, data, "application/pdf")}, data={"course": course})
        if r.status_code != 202:
            raise RuntimeError(r.json().get("detail", r.text))
        return r.json()

    def task(self, task_id: str) -> dict[str, Any]:
        return self.http.get(f"/documents/tasks/{task_id}").raise_for_status().json()

    def documents(self, course: str | None) -> list[dict[str, Any]]:
        return self.http.get("/documents", params={"course": course} if course else None).raise_for_status().json()

    def delete(self, file_id: str) -> None:
        self.http.delete(f"/documents/{file_id}").raise_for_status()

    def feedback(self, **body: Any) -> None:
        self.http.post("/feedback", json=body).raise_for_status()

    def chat(self, message: str, thread_id: str, user_id: str, course: str | None) -> Iterator[tuple[str, dict]]:
        body = {"message": message, "thread_id": thread_id, "user_id": user_id, "course": course}
        yield from self._stream("/chat", body)

    def resume(self, answers: list[int | None], thread_id: str, user_id: str, course: str | None):
        body = {"answers": answers, "thread_id": thread_id, "user_id": user_id, "course": course}
        yield from self._stream("/chat/resume", body)

    def _stream(self, path: str, body: dict) -> Iterator[tuple[str, dict]]:
        # ответ агента может идти минуту (реранкер на CPU, лимиты провайдера) — без таймаута на чтение
        timeout = httpx.Timeout(None, connect=5)
        with self.http.stream("POST", path, json=body, timeout=timeout) as r:
            r.raise_for_status()
            event = "message"
            for line in r.iter_lines():
                if line.startswith("event:"):
                    event = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    yield event, json.loads(line.split(":", 1)[1])
                    event = "message"
