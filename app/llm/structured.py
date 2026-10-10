"""Структурированный вывод: инструкция с примером объекта и разбор JSON из ответа модели.

Модели показываем не JSON Schema, а пример объекта нужной формы. На голую схему GigaChat
отвечал самой схемой — `{"properties": {...}, "type": "object"}` — и каждый структурированный
ответ требовал второго вызова на починку.
"""

import json
import re
import types
from typing import Any, Literal, TypeVar, Union, get_args, get_origin

from pydantic import BaseModel, ValidationError

M = TypeVar("M", bound=BaseModel)

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def json_instruction(schema: type[BaseModel]) -> str:
    example = json.dumps(example_object(schema), ensure_ascii=False, indent=2)
    notes = []
    for name, f in _all_fields(schema):
        optional = not f.is_required() and f.default is None
        if f.description or optional:
            notes.append(f"- {name}: " + "; ".join(filter(None, [f.description, "может быть null" if optional else ""])))
    text = (
        "Ответь ТОЛЬКО одним JSON-объектом, без пояснений и без markdown. Форма объекта — как в примере; "
        "подставь свои значения, а где варианты перечислены через « | », выбери ровно один:\n" + example
    )
    if notes:
        text += "\nПояснения к полям:\n" + "\n".join(notes)
    return text


def example_object(schema: type[BaseModel]) -> dict[str, Any]:
    return {name: _example(f.annotation, f.description) for name, f in schema.model_fields.items()}


def _example(annotation: Any, description: str | None = None) -> Any:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Literal:
        return " | ".join(str(a) for a in args)
    if origin in (Union, types.UnionType):
        non_null = [a for a in args if a is not type(None)]
        return _example(non_null[0], description) if non_null else None
    if origin in (list, tuple, set):
        return [_example(args[0])] if args else []
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return example_object(annotation)
    if annotation is bool:
        return True
    if annotation is int:
        return 0
    if annotation is float:
        return 0.0
    return "…"


def _all_fields(schema: type[BaseModel], prefix: str = ""):
    for name, f in schema.model_fields.items():
        yield prefix + name, f
        inner = [a for a in (f.annotation, *get_args(f.annotation)) if isinstance(a, type) and issubclass(a, BaseModel)]
        for model in inner:
            yield from _all_fields(model, prefix=f"{prefix}{name}[].")


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
