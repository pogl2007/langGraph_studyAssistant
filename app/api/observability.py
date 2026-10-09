"""Структурные логи в JSON и request_id, который проходит через весь запрос.

request_id лежит в contextvar: его видит любой логгер внутри запроса — в том числе в узлах графа
и в слое LLM, — без протаскивания через аргументы. Middleware написан на чистом ASGI, а не через
BaseHTTPMiddleware: тот плохо дружит со стриминговыми ответами (SSE).
"""

import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar

request_id: ContextVar[str] = ContextVar("request_id", default="-")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "request_id": request_id.get(),
            "msg": record.getMessage(),
        }
        if record.exc_info:
            data["exc"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False)


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    for noisy in ("httpx", "httpcore", "sentence_transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class RequestIdMiddleware:
    def __init__(self, app):
        self.app = app
        self.log = logging.getLogger("app.api.access")

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        incoming = dict(scope.get("headers") or {}).get(b"x-request-id", b"").decode()[:64]
        rid = incoming or uuid.uuid4().hex[:12]
        token = request_id.set(rid)
        started = time.perf_counter()
        status = 500

        async def send_with_id(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message.setdefault("headers", []).append((b"x-request-id", rid.encode()))
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            self.log.info(
                "%s %s -> %d %.0fms", scope["method"], scope["path"], status, (time.perf_counter() - started) * 1000
            )
            request_id.reset(token)
