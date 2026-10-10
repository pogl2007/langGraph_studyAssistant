"""Адаптер GigaChat (Сбер) к общему интерфейсу провайдера.

Почему не через OpenAI-совместимый Provider:
- авторизация OAuth: ключ авторизации (base64 от client_id:client_secret) обменивается на access
  token на 30 минут на отдельном хосте ngw.devices.sberbank.ru — SDK `gigachat` обновляет токен сам;
- TLS-сертификаты подписаны корневым сертификатом Минцифры, которого нет в стандартных наборах
  Python, — нужен `GIGACHAT_CA_BUNDLE` или явно выключенная проверка;
- у SDK свои исключения — здесь они переводятся в общие типы, и политика ретраев и fallback
  в LLMClient работает для GigaChat так же, как для остальных.
"""

import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.config import Settings
from app.llm.client import Completion, EmptyCompletion, LLMError, RateLimited, RejectedError, TransientError

log = logging.getLogger(__name__)


class GigaChatProvider:
    def __init__(
        self,
        credentials: str,
        model: str,
        scope: str = "GIGACHAT_API_PERS",
        base_url: str = "",
        auth_url: str = "",
        verify_ssl: bool = True,
        ca_bundle: str = "",
        timeout: float = 60.0,
        name: str = "gigachat",
        client: Any = None,
    ):
        self.name = name
        self.model = model
        if client is None:
            from gigachat import GigaChat

            kwargs: dict[str, Any] = {
                "credentials": credentials,
                "scope": scope,
                "model": model,
                "timeout": timeout,
                "max_retries": 0,  # повторы решает LLMClient
                "verify_ssl_certs": verify_ssl,
            }
            if base_url:
                kwargs["base_url"] = base_url
            if auth_url:
                kwargs["auth_url"] = auth_url
            if ca_bundle:
                kwargs["ca_bundle_file"] = ca_bundle
            if not verify_ssl:
                log.warning("gigachat: проверка TLS-сертификата выключена — укажите GIGACHAT_CA_BUNDLE")
            client = GigaChat(**kwargs)
        self.client = client

    @classmethod
    def from_settings(cls, s: Settings) -> "GigaChatProvider":
        return cls(
            credentials=s.gigachat_credentials.get_secret_value(),
            model=s.gigachat_model,
            scope=s.gigachat_scope,
            base_url=s.gigachat_base_url,
            auth_url=s.gigachat_auth_url,
            verify_ssl=s.gigachat_verify_ssl,
            ca_bundle=s.gigachat_ca_bundle,
            timeout=s.llm_timeout,
        )

    def _payload(self, messages: list[dict], max_tokens: int, temperature: float) -> dict[str, Any]:
        return {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}

    async def complete(self, messages: list[dict], max_tokens: int, temperature: float) -> Completion:
        started = time.perf_counter()
        try:
            response = await self.client.achat(self._payload(messages, max_tokens, temperature))
        except Exception as e:
            _reraise(e)
        text = (response.choices[0].message.content or "").strip() if response.choices else ""
        usage = response.usage
        completion = Completion(
            text=text,
            model=getattr(response, "model", None) or self.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_s=time.perf_counter() - started,
            provider=self.name,
        )
        if not text:
            raise EmptyCompletion(f"пустой ответ, out={completion.output_tokens}")
        return completion

    async def stream(self, messages: list[dict], max_tokens: int, temperature: float) -> AsyncIterator[str | Completion]:
        started = time.perf_counter()
        parts: list[str] = []
        usage = None
        model = self.model
        try:
            async for chunk in self.client.astream(self._payload(messages, max_tokens, temperature)):
                usage = getattr(chunk, "usage", None) or usage
                model = getattr(chunk, "model", None) or model
                if chunk.choices and chunk.choices[0].delta.content:
                    parts.append(chunk.choices[0].delta.content)
                    yield chunk.choices[0].delta.content
        except Exception as e:
            _reraise(e)
        text = "".join(parts).strip()
        if not text:
            raise EmptyCompletion("пустой ответ в стриме")
        yield Completion(
            text=text,
            model=model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_s=time.perf_counter() - started,
            provider=self.name,
        )

    async def aclose(self) -> None:
        await self.client.aclose()


def _translate(e: Exception) -> Exception:
    """Ошибки SDK gigachat → общие типы. Чужое исключение возвращается как есть:
    это баг в нашем коде, и LLMClient не должен прятать его за fallback."""
    from gigachat import exceptions as gx

    if isinstance(e, LLMError):
        return e
    if isinstance(e, gx.RateLimitError):
        return RateLimited(_describe(e), e.retry_after or None)
    if isinstance(e, gx.ServerError):
        return TransientError(_describe(e))
    if isinstance(e, gx.ResponseError):  # 400, 401, 403, 404, 413, 422
        return RejectedError(_describe(e), e.status_code)
    if isinstance(e, (httpx.TimeoutException, httpx.TransportError)):
        return TransientError(f"{type(e).__name__}: {e}")
    if isinstance(e, gx.GigaChatException):
        return RejectedError(f"{type(e).__name__}: {e}")
    return e


def _reraise(e: Exception) -> None:
    translated = _translate(e)
    if translated is e:
        raise e
    raise translated from e


def _describe(e) -> str:
    content = (e.content or b"")[:200].decode("utf-8", "replace")
    return f"gigachat {e.status_code}: {content}"
