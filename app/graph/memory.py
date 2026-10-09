"""Долговременная память: темы, в которых студент ошибается в тестах.

Короткая память (история диалога) живёт в checkpointer'е и привязана к thread_id — новый диалог
начинается с чистого листа. Слабые темы относятся к человеку, а не к диалогу, поэтому лежат в
LangGraph Store под пространством ("users", user_id, "weak_topics") и переживают любые диалоги.

Ошибка увеличивает счётчик темы, правильный ответ уменьшает. Дошёл до нуля — тема освоена и удаляется.
"""

import re
from datetime import UTC, datetime
from typing import Any

from langgraph.store.base import BaseStore


def namespace(user_id: str) -> tuple[str, ...]:
    return ("users", user_id, "weak_topics")


def topic_key(topic: str) -> str:
    key = re.sub(r"\s+", " ", topic.lower().replace("ё", "е")).strip(" .,;:«»\"'")
    return key[:100] or "без темы"


async def record_answers(store: BaseStore, user_id: str, results: list[dict[str, Any]]) -> list[str]:
    """results: [{"topic", "correct": bool, "source": {...}}]. Возвращает темы, ставшие слабыми."""
    ns = namespace(user_id)
    now = datetime.now(UTC).isoformat(timespec="seconds")
    weakened: list[str] = []
    for r in results:
        key = topic_key(r["topic"])
        item = await store.aget(ns, key)
        if r["correct"]:
            if item is None:
                continue
            value = dict(item.value)
            value["correct"] = value.get("correct", 0) + 1
            value["mistakes"] = max(0, value.get("mistakes", 0) - 1)
            value["last_seen"] = now
            if value["mistakes"] == 0:
                await store.adelete(ns, key)  # тема освоена
            else:
                await store.aput(ns, key, value)
        else:
            value = dict(item.value) if item else {"topic": r["topic"], "mistakes": 0, "correct": 0, "sources": []}
            value["mistakes"] = value.get("mistakes", 0) + 1
            value["last_seen"] = now
            source = r.get("source")
            if source and source not in value["sources"]:
                value["sources"] = [*value["sources"], source][-3:]
            await store.aput(ns, key, value)
            weakened.append(value["topic"])
    return weakened


async def weak_topics(store: BaseStore | None, user_id: str, limit: int = 10) -> list[dict[str, Any]]:
    if store is None:
        return []
    items = await store.asearch(namespace(user_id), limit=100)
    values = [i.value for i in items if i.value.get("mistakes", 0) > 0]
    # больше ошибок — выше; при равенстве свежие раньше (сортировка устойчивая)
    values.sort(key=lambda v: v.get("last_seen", ""), reverse=True)
    values.sort(key=lambda v: -v["mistakes"])
    return values[:limit]
