"""Адаптер GigaChat: ошибки SDK переводятся в общие типы, и политика ретраев/fallback работает так же."""

from types import SimpleNamespace

import httpx
import pytest
from gigachat import exceptions as gx

from app.llm.client import AllProvidersFailed, Completion, LLMClient
from app.llm.gigachat import GigaChatProvider

from .conftest import make_settings

URL = "https://api.giga.chat/v1/chat/completions"


def reply(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=4),
        model="GigaChat-3-Ultra",
    )


def chunk(text: str | None, usage=None) -> SimpleNamespace:
    choices = [SimpleNamespace(delta=SimpleNamespace(content=text))] if text is not None else []
    return SimpleNamespace(choices=choices, usage=usage, model="GigaChat-3-Ultra")


class FakeSDK:
    """Подменяет клиента gigachat: отдаёт заготовленные ответы или бросает исключения SDK."""

    def __init__(self, *script):
        self.script = list(script)
        self.payloads: list[dict] = []

    async def achat(self, payload):
        self.payloads.append(payload)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def astream(self, payload):
        self.payloads.append(payload)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        for c in item:
            yield c

    async def aclose(self):
        pass


def provider(*script, name="gigachat") -> GigaChatProvider:
    return GigaChatProvider(credentials="x", model="GigaChat-3-Ultra", name=name, client=FakeSDK(*script))


def client(*providers, sleeps=None) -> LLMClient:
    async def fake_sleep(s):
        if sleeps is not None:
            sleeps.append(s)

    return LLMClient(list(providers), max_retries=2, max_retry_wait=8, sleep=fake_sleep)


MSG = [{"role": "system", "content": "кратко"}, {"role": "user", "content": "привет"}]


async def test_complete_maps_usage_and_payload():
    p = provider(reply("Привет!"))
    c = await client(p).complete(MSG, max_tokens=100, temperature=0.0)
    assert (c.text, c.provider, c.input_tokens, c.output_tokens) == ("Привет!", "gigachat", 11, 4)
    assert p.client.payloads[0] == {"model": "GigaChat-3-Ultra", "messages": MSG, "max_tokens": 100, "temperature": 0.0}


async def test_rate_limit_with_short_retry_after_is_retried():
    sleeps = []
    limited = gx.RateLimitError(URL, 429, b"too many", httpx.Headers({"retry-after": "3"}))
    c = await client(provider(limited, reply("ok")), sleeps=sleeps).complete(MSG)
    assert c.text == "ok"
    assert sleeps == [3.0]


async def test_rate_limit_without_retry_after_backs_off_briefly():
    # так GigaChat отвечает на второй параллельный запрос с того же ключа
    sleeps = []
    busy = gx.RateLimitError(URL, 429, b'{"message":"Too Many Requests"}', None)
    c = await client(provider(busy, busy, reply("ok")), sleeps=sleeps).complete(MSG)
    assert c.provider == "gigachat"
    assert sleeps == [1.0, 2.0]


async def test_auth_error_falls_back_to_next_provider():
    gigachat = provider(gx.AuthenticationError(URL, 401, b"bad credentials", None))
    backup = provider(reply("ответ запасного"), name="backup")
    c = await client(gigachat, backup).complete(MSG)
    assert c.provider == "backup"


async def test_server_error_and_timeout_are_retried_with_backoff():
    sleeps = []
    p = provider(gx.ServerError(URL, 503, b"", None), httpx.ConnectTimeout("timeout"), reply("ok"))
    c = await client(p, sleeps=sleeps).complete(MSG)
    assert c.text == "ok"
    assert len(sleeps) == 2 and sleeps[1] > sleeps[0]


async def test_all_failed_lists_gigachat_error():
    with pytest.raises(AllProvidersFailed) as e:
        await client(provider(gx.ForbiddenError(URL, 403, b"scope", None))).complete(MSG)
    assert "gigachat 403" in str(e.value)


async def test_bug_in_our_code_is_not_hidden_by_fallback():
    with pytest.raises(KeyError):
        await client(provider(KeyError("oops")), provider(reply("ok"), name="backup")).complete(MSG)


async def test_stream_tokens_and_final_usage():
    usage = SimpleNamespace(prompt_tokens=20, completion_tokens=3)
    p = provider([chunk("При"), chunk("вет"), chunk(None, usage)])
    items = [i async for i in client(p).stream(MSG)]
    assert items[:2] == ["При", "вет"]
    assert isinstance(items[-1], Completion)
    assert (items[-1].text, items[-1].output_tokens) == ("Привет", 3)


async def test_empty_answer_falls_back():
    c = await client(provider(reply("")), provider(reply("ok"), name="backup")).complete(MSG)
    assert c.provider == "backup"


def test_chain_order_comes_from_settings():
    s = make_settings(llm_chain="gigachat,openai", gigachat_credentials="x", llm_fallback_models="m2")
    names = [p.name for p in LLMClient.from_settings(s).providers]
    assert names == ["gigachat", "primary", "primary:m2"]


def test_gigachat_is_skipped_without_credentials():
    s = make_settings(llm_chain="gigachat,openai", gigachat_credentials="")
    assert [p.name for p in LLMClient.from_settings(s).providers] == ["primary"]


def test_ca_bundle_path_is_resolved_from_project_root():
    from app.config import ROOT
    from app.llm.gigachat import resolve_ca_bundle

    path = resolve_ca_bundle("certs/russian_trusted_root_ca.pem")
    assert path == str(ROOT / "certs" / "russian_trusted_root_ca.pem")
    assert resolve_ca_bundle("") == ""


def test_missing_ca_bundle_fails_at_startup_not_on_first_request():
    from app.llm.gigachat import resolve_ca_bundle

    with pytest.raises(ValueError, match="не найден"):
        resolve_ca_bundle("certs/nope.pem")


class FakeImageSDK:
    def __init__(self, content: str, image_b64: str = ""):
        self.content = content
        self.image_b64 = image_b64
        self.payloads: list[dict] = []
        self.fetched: list[str] = []

    async def achat(self, payload):
        self.payloads.append(payload)
        return reply(self.content)

    async def aget_image(self, file_id):
        self.fetched.append(file_id)
        return SimpleNamespace(content=self.image_b64)

    async def aclose(self):
        pass


async def test_image_generator_parses_img_tag_and_downloads_file():
    import base64

    from app.llm.images import GigaChatImageGenerator

    sdk = FakeImageSDK('<img src="abc-123" fuse="true"/>Схема готова', base64.b64encode(b"JPEG").decode())
    image = await GigaChatImageGenerator(sdk, "GigaChat-3-Ultra").generate("нарисуй поле")
    assert image.data == b"JPEG"
    assert image.caption == "Схема готова"
    assert sdk.fetched == ["abc-123"]
    assert sdk.payloads[0]["function_call"] == "auto"  # без этого модель не вызывает рисование


async def test_image_generator_text_answer_is_a_rejection():
    from app.llm.client import RejectedError
    from app.llm.images import GigaChatImageGenerator

    with pytest.raises(RejectedError, match="не вызвала рисование"):
        await GigaChatImageGenerator(FakeImageSDK("Я не могу рисовать"), "m").generate("нарисуй")
