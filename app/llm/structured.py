"""Структурированный вывод: инструкция со схемой и разбор JSON из ответа модели."""

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

M = TypeVar("M", bound=BaseModel)

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def json_instruction(schema: type[BaseModel]) -> str:
    return (
        "Ответь ТОЛЬКО одним JSON-объектом без пояснений и без markdown, строго по схеме:\n"
        + json.dumps(schema.model_json_schema(), ensure_ascii=False)
    )


def parse_json(text: str, schema: type[M]) -> M:
    """Достаёт JSON из ответа (модели любят обернуть его в ```json или добавить фразу) и проверяет схему."""
    cleaned = _FENCE.sub("", text.strip())
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"в ответе нет JSON-объекта: {text[:200]!r}")
    try:
        return schema.model_validate_json(cleaned[start : end + 1])
    except ValidationError as e:
        raise ValueError(str(e)) from e
