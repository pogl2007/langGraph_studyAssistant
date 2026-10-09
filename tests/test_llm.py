"""Слой LLM: ретраи, fallback между провайдерами, стриминг и структурированный вывод.

HTTP подменён httpx.MockTransport — настоящий SDK openai, настоящий разбор ответов, без сети."""

import json

import httpx
import pytest
from pydantic import BaseModel

from app.llm.client import AllProvidersFailed, Completion, LLMClient, Provider, retry_after


def chat_response(text: str, model: str = "m") -> httpx.Response:
    return httpx.Response(200, json={
        "id": "x", "object": "chat.completion", "created": 0, "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
    })


def stream_response(tokens: list[str], model: str = "m") -> httpx.Response:
    events = [
        {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": model,
         "choices": [{"index": 0, "delta": {"content": t}, "finish_reason": None}]}
        for t in tokens
    ]
    events.append({"id": "x", "object": "chat.completion.chunk", "created": 0, "model": model, "choices": [],
                   "usage": {"prompt_tokens": 7, "completion_tokens": len(tokens), "total_tokens": 7 + len(tokens)}})
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
    return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})


def error(status: int, message: str = "err", headers: dict | None = None) -> httpx.Response:
    return httpx.Response(status, json={"error": {"message": message}}, headers=headers or {})


class Script:
    """По модели в теле запроса отдаёт следующий заготовленный ответ."""

    def __init__(self, **by_model: list[httpx.Response]):
        self.by_model = by_model
        self.seen: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        self.seen.append(model)
        return self.by_model[model].pop(0)


def client(script: Script, *models: str, sleeps: list | None = None, **kw) -> LLMClient:
    http = httpx.AsyncClient(transport=httpx.MockTransport(script))
    providers = [Provider(m, "http://llm.test/v1", "k", m, timeout=5, http_client=http) for m in models]

    async def fake_sleep(s: float) -> None:
        if sleeps is not None:
            sleeps.append(s)

    return LLMClient(providers, max_retries=kw.pop("max_retries", 2), max_retry_wait=8, sleep=fake_sleep, **kw)


MSG = [{"role": "user", "content": "привет"}]


async def test_success_logs_usage():
    c = await client(Script(a=[chat_response("ok")]), "a").complete(MSG)
    assert (c.text, c.provider, c.input_tokens, c.output_tokens) == ("ok", "a", 7, 3)


async def test_short_retry_after_waits_and_retries_same_provider():
    sleeps = []
    script = Script(a=[error(429, headers={"retry-after": "2"}), chat_response("ok")])
    c = await client(script, "a", "b", sleeps=sleeps).complete(MSG)
    assert c.provider == "a"
    assert sleeps == [2.0]
    assert script.seen == ["a", "a"]


async def test_long_retry_after_goes_to_next_provider_without_waiting():
    sleeps = []
    script = Script(a=[error(429, "Rate limit exceeded. Retry in 56s.")], b=[chat_response("ok")])
    c = await client(script, "a", "b", sleeps=sleeps).complete(MSG)
    assert c.provider == "b"
    assert sleeps == []


async def test_payment_required_is_not_retried():
    script = Script(a=[error(402, "subscription required")], b=[chat_response("ok")])
    c = await client(script, "a", "b").complete(MSG)
    assert script.seen == ["a", "b"]
    assert c.text == "ok"


async def test_server_errors_back_off_then_fall_back():
    sleeps = []
    script = Script(a=[error(500), error(502), error(503)], b=[chat_response("ok")])
    c = await client(script, "a", "b", sleeps=sleeps).complete(MSG)
    assert c.provider == "b"
    assert len(sleeps) == 2 and sleeps[1] > sleeps[0]  # экспоненциальная задержка


async def test_empty_answer_falls_back():
    script = Script(a=[chat_response("")], b=[chat_response("ok")])
    assert (await client(script, "a", "b").complete(MSG)).provider == "b"


async def test_all_failed_reports_every_provider():
    script = Script(a=[error(401)], b=[error(404)])
    with pytest.raises(AllProvidersFailed) as e:
        await client(script, "a", "b").complete(MSG)
    assert len(e.value.errors) == 2


async def test_stream_yields_tokens_then_completion():
    items = [i async for i in client(Script(a=[stream_response(["При", "вет"])]), "a").stream(MSG)]
    assert items[:2] == ["При", "вет"]
    assert isinstance(items[-1], Completion)
    assert items[-1].text == "Привет"
    assert items[-1].output_tokens == 2


async def test_stream_falls_back_before_first_token():
    script = Script(a=[error(402)], b=[stream_response(["ok"])])
    items = [i async for i in client(script, "a", "b").stream(MSG)]
    assert items[-1].provider == "b"


class Answer(BaseModel):
    route: str
    score: int


async def test_json_is_parsed_from_fenced_answer():
    script = Script(a=[chat_response('Вот ответ:\n```json\n{"route": "quiz", "score": 3}\n```')])
    parsed, _ = await client(script, "a").complete_json(MSG, Answer)
    assert parsed == Answer(route="quiz", score=3)


async def test_invalid_json_gets_one_repair_attempt():
    script = Script(a=[chat_response('{"route": "quiz"}'), chat_response('{"route": "quiz", "score": 1}')])
    parsed, _ = await client(script, "a").complete_json(MSG, Answer)
    assert parsed.score == 1
    assert len(script.seen) == 2


def test_retry_after_parsing():
    class E(Exception):
        response = None

    assert retry_after(E("Rate limit exceeded. Retry in 12s.")) == 12.0
    assert retry_after(E("nothing here")) is None
