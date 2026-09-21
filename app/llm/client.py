"""Клиент LLM через OpenAI-совместимый API.

Один интерфейс для Ollama, vLLM и внешних провайдеров: меняются только base_url, ключ и модель.
Адаптеры, ретраи и fallback между провайдерами — этап 6.
"""

import logging
import time
from dataclasses import dataclass

from openai import AsyncOpenAI

from app.config import Settings

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float


class LLMClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float,
        max_tokens: int,
        temperature: float,
        max_retries: int = 1,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=max_retries)

    @classmethod
    def from_settings(cls, s: Settings) -> "LLMClient":
        return cls(
            base_url=s.llm_base_url,
            api_key=s.llm_api_key.get_secret_value(),
            model=s.llm_model,
            timeout=s.llm_timeout,
            max_tokens=s.llm_max_tokens,
            temperature=s.llm_temperature,
        )

    async def complete(self, messages: list[dict[str, str]]) -> Completion:
        started = time.perf_counter()
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        except Exception as e:
            log.warning("llm model=%s error=%s latency=%.2fs", self.model, e, time.perf_counter() - started)
            raise
        latency = time.perf_counter() - started
        usage = response.usage
        result = Completion(
            text=(response.choices[0].message.content or "").strip(),
            model=response.model or self.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_s=latency,
        )
        log.info(
            "llm model=%s in=%d out=%d latency=%.2fs",
            result.model, result.input_tokens, result.output_tokens, latency,
        )
        return result

    async def aclose(self) -> None:
        await self._client.close()
