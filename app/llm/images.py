"""Генерация картинок через GigaChat (встроенная функция text2image, модель Kandinsky).

Отдельного API у GigaChat для этого нет: это обычный chat-запрос с `function_call: "auto"`.
Модель сама решает вызвать функцию рисования и возвращает в тексте тег
`<img src="<file_id>" fuse="true"/>`, а файл скачивается отдельно по file_id.

Fallback-цепочки здесь нет: у запасных OpenAI-совместимых провайдеров генерации картинок нет.
"""

import base64
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol

from app.config import Settings
from app.llm.client import LLMError, RejectedError

log = logging.getLogger(__name__)

_IMG = re.compile(r'<img\s+src="([^"]+)"[^>]*/?>')

IMAGE_SYSTEM = (
    "Ты рисуешь учебные иллюстрации к лекциям. Простая чёткая схема на белом фоне, как в учебнике: "
    "без подписей, букв и формул — текст на картинках получается нечитаемым."
)


@dataclass(frozen=True)
class GeneratedImage:
    data: bytes
    caption: str  # что модель написала рядом с картинкой
    latency_s: float


class ImageGenerator(Protocol):
    async def generate(self, prompt: str) -> GeneratedImage: ...


class GigaChatImageGenerator:
    def __init__(self, client: Any, model: str):
        self.client = client
        self.model = model

    @classmethod
    def from_settings(cls, s: Settings) -> "GigaChatImageGenerator":
        from gigachat import GigaChat

        from app.llm.gigachat import resolve_ca_bundle

        kwargs: dict[str, Any] = {
            "credentials": s.gigachat_credentials.get_secret_value(),
            "scope": s.gigachat_scope,
            "model": s.gigachat_model,
            "timeout": max(s.llm_timeout, 120.0),  # рисование заметно дольше текста
            "max_retries": 0,
            "verify_ssl_certs": s.gigachat_verify_ssl,
        }
        if s.gigachat_base_url:
            kwargs["base_url"] = s.gigachat_base_url
        if s.gigachat_auth_url:
            kwargs["auth_url"] = s.gigachat_auth_url
        if s.gigachat_verify_ssl and s.gigachat_ca_bundle:
            kwargs["ca_bundle_file"] = resolve_ca_bundle(s.gigachat_ca_bundle)
        return cls(GigaChat(**kwargs), s.gigachat_model)

    async def generate(self, prompt: str) -> GeneratedImage:
        from app.llm.gigachat import _reraise

        started = time.perf_counter()
        try:
            response = await self.client.achat({
                "model": self.model,
                "function_call": "auto",
                "messages": [{"role": "system", "content": IMAGE_SYSTEM}, {"role": "user", "content": prompt}],
            })
            content = response.choices[0].message.content or ""
            match = _IMG.search(content)
            if not match:
                # Модель ответила текстом и не стала рисовать — для пользователя это отказ, а не сбой сети
                raise RejectedError(f"модель не вызвала рисование: {content[:150]!r}")
            image = await self.client.aget_image(match.group(1))
        except LLMError:
            raise
        except Exception as e:
            _reraise(e)
        data = base64.b64decode(image.content)
        caption = _IMG.sub("", content).strip()
        latency = time.perf_counter() - started
        log.info("image provider=gigachat model=%s bytes=%d latency=%.1fs", self.model, len(data), latency)
        return GeneratedImage(data=data, caption=caption, latency_s=latency)

    async def aclose(self) -> None:
        await self.client.aclose()
