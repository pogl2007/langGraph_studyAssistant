"""Клиент LLM поверх OpenAI-совместимого API с fallback между провайдерами.

Ollama, vLLM и внешние провайдеры говорят на одном протоколе `/v1/chat/completions`, поэтому
провайдер — это тройка (base_url, ключ, модель). Цепочка провайдеров задаётся в конфиге.

Политика ошибок:
- 429 с коротким Retry-After — ждём и повторяем у того же провайдера; с длинным — сразу к следующему:
  минута ожидания хуже, чем ответ запасной модели;
- таймаут, обрыв соединения, 5xx — экспоненциальная задержка, потом следующий провайдер;
- прочие 4xx (402 «нужна подписка», 401, 404) — повтор не поможет, сразу следующий;
- пустой ответ (рассуждающая модель съела лимит токенов) — тоже повод взять следующего.
"""

import asyncio
import logging
import random
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)

from app.config import Settings

log = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    provider: str = ""


class LLMError(Exception):
    pass


class EmptyCompletion(LLMError):
    pass


@dataclass
class AllProvidersFailed(LLMError):
    errors: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return "все LLM-провайдеры недоступны: " + "; ".join(self.errors)


class Provider:
    def __init__(self, name: str, base_url: str, api_key: str, model: str, timeout: float, http_client=None):
        self.name = name
        self.model = model
        # Свои ретраи SDK выключены: повторы и fallback решает LLMClient
        self.client = AsyncOpenAI(
            base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0, http_client=http_client
        )

    async def complete(self, messages: list[dict], max_tokens: int, temperature: float) -> Completion:
        started = time.perf_counter()
        response = await self.client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=max_tokens, temperature=temperature
        )
        text = (response.choices[0].message.content or "").strip() if response.choices else ""
        usage = response.usage
        completion = Completion(
            text=text,
            model=response.model or self.model,
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
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True},
        )
        parts: list[str] = []
        usage = None
        model = self.model
        async for chunk in stream:
            usage = chunk.usage or usage
            model = chunk.model or model
            if chunk.choices and chunk.choices[0].delta.content:
                parts.append(chunk.choices[0].delta.content)
                yield chunk.choices[0].delta.content
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
        await self.client.close()


class LLMClient:
    def __init__(
        self,
        providers: list[Provider],
        max_tokens: int = 2048,
        temperature: float = 0.1,
        max_retries: int = 2,
        max_retry_wait: float = 8.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        if not providers:
            raise ValueError("нужен хотя бы один провайдер")
        self.providers = providers
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.max_retry_wait = max_retry_wait
        self._sleep = sleep

    @property
    def model(self) -> str:
        return self.providers[0].model

    @classmethod
    def from_settings(cls, s: Settings) -> "LLMClient":
        key = s.llm_api_key.get_secret_value()
        providers = [Provider(s.llm_provider_name, s.llm_base_url, key, s.llm_model, s.llm_timeout)]
        for model in filter(None, (m.strip() for m in s.llm_fallback_models.split(","))):
            providers.append(Provider(f"{s.llm_provider_name}:{model}", s.llm_base_url, key, model, s.llm_timeout))
        if s.llm_fallback_base_url and s.llm_fallback_model:
            providers.append(
                Provider("fallback", s.llm_fallback_base_url, s.llm_fallback_api_key.get_secret_value(),
                         s.llm_fallback_model, s.llm_timeout)
            )
        return cls(providers, s.llm_max_tokens, s.llm_temperature, s.llm_max_retries, s.llm_max_retry_wait)

    async def complete(
        self, messages: list[dict], max_tokens: int | None = None, temperature: float | None = None
    ) -> Completion:
        max_tokens = max_tokens or self.max_tokens
        temperature = self.temperature if temperature is None else temperature
        return await self._with_fallback(lambda p: p.complete(messages, max_tokens, temperature))

    async def stream(
        self, messages: list[dict], max_tokens: int | None = None, temperature: float | None = None
    ) -> AsyncIterator[str | Completion]:
        """Отдаёт токены по мере генерации, последним элементом — Completion с usage.

        Переключиться на другого провайдера можно только до первого токена: ответ,
        склеенный из двух моделей, пользователю показывать нельзя."""
        max_tokens = max_tokens or self.max_tokens
        temperature = self.temperature if temperature is None else temperature
        errors: list[str] = []
        for provider in self.providers:
            for attempt in range(self.max_retries + 1):
                started = False
                try:
                    async for item in provider.stream(messages, max_tokens, temperature):
                        started = True
                        if isinstance(item, Completion):
                            _log_ok(item)
                        yield item
                    return
                except Exception as e:
                    if started:
                        raise
                    action = await self._on_error(provider, attempt, e, errors)
                    if action == "next":
                        break
        raise AllProvidersFailed(errors)

    async def complete_json(
        self, messages: list[dict], schema: type[T], max_tokens: int | None = None
    ) -> tuple[T, Completion]:
        """Структурированный ответ: JSON по схеме Pydantic, с одной попыткой починки.

        Не полагаемся на response_format: json_schema поддерживают не все OpenAI-совместимые
        серверы, а проверка Pydantic работает одинаково для любого провайдера."""
        from app.llm.structured import json_instruction, parse_json

        prompt = [*messages, {"role": "user", "content": json_instruction(schema)}]
        completion = await self.complete(prompt, max_tokens=max_tokens, temperature=0.0)
        try:
            return parse_json(completion.text, schema), completion
        except ValueError as e:
            log.warning("llm json invalid, repair: %s", str(e)[:200])
            repair = [
                *prompt,
                {"role": "assistant", "content": completion.text},
                {"role": "user", "content": f"Ответ не прошёл проверку: {str(e)[:500]}. Верни только исправленный JSON."},
            ]
            completion = await self.complete(repair, max_tokens=max_tokens, temperature=0.0)
            return parse_json(completion.text, schema), completion

    async def _with_fallback(self, call: Callable[[Provider], Awaitable[Completion]]) -> Completion:
        errors: list[str] = []
        for provider in self.providers:
            for attempt in range(self.max_retries + 1):
                try:
                    completion = await call(provider)
                    _log_ok(completion)
                    return completion
                except Exception as e:
                    if await self._on_error(provider, attempt, e, errors) == "next":
                        break
        raise AllProvidersFailed(errors)

    async def _on_error(self, provider: Provider, attempt: int, error: Exception, errors: list[str]) -> str:
        """Решает, повторить у этого провайдера ("retry") или перейти к следующему ("next")."""
        last_try = attempt >= self.max_retries
        if isinstance(error, RateLimitError):
            wait = retry_after(error)
            if wait is not None and wait <= self.max_retry_wait and not last_try:
                return await self._retry(provider, attempt, error, wait)
        elif isinstance(error, (APITimeoutError, APIConnectionError, InternalServerError)) and not last_try:
            return await self._retry(provider, attempt, error, 0.5 * 2**attempt + random.uniform(0, 0.3))
        elif not isinstance(error, (APIStatusError, EmptyCompletion, APITimeoutError, APIConnectionError)):
            raise error  # ошибка в нашем коде, а не у провайдера — fallback её только спрячет
        errors.append(f"{provider.name}: {_short(error)}")
        log.warning("llm provider=%s model=%s error=%s -> next provider", provider.name, provider.model, _short(error))
        return "next"

    async def _retry(self, provider: Provider, attempt: int, error: Exception, wait: float) -> str:
        log.warning(
            "llm provider=%s model=%s attempt=%d error=%s retry_in=%.1fs",
            provider.name, provider.model, attempt + 1, _short(error), wait,
        )
        await self._sleep(wait)
        return "retry"

    async def aclose(self) -> None:
        for p in self.providers:
            await p.aclose()


_RETRY_IN = re.compile(r"[Rr]etry in (\d+(?:\.\d+)?)\s*s")


def retry_after(error: Any) -> float | None:
    """Сколько ждать по ответу 429: заголовок Retry-After или текст «Retry in 12s» (apinex)."""
    response = getattr(error, "response", None)
    header = response.headers.get("retry-after") if response is not None else None
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    m = _RETRY_IN.search(str(error))
    return float(m.group(1)) if m else None


def _short(error: Exception) -> str:
    status = getattr(error, "status_code", None)
    text = str(error).replace("\n", " ")[:160]
    return f"{type(error).__name__}({status}) {text}" if status else f"{type(error).__name__} {text}"


def _log_ok(c: Completion) -> None:
    log.info(
        "llm provider=%s model=%s in=%d out=%d latency=%.2fs",
        c.provider, c.model, c.input_tokens, c.output_tokens, c.latency_s,
    )
