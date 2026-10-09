import hashlib
import math
import re
from collections.abc import Callable, Sequence
from typing import Any

import pymupdf
import pytest
from qdrant_client import QdrantClient, models

from app.config import Settings
from app.ingest.indexer import Indexer
from app.llm.client import Completion, LLMError
from app.llm.structured import parse_json
from app.retrieval.search import Retriever
from app.retrieval.sparse import normalize
from app.retrieval.store import ChunkStore

_WORD = re.compile(r"\w+", re.UNICODE)


def _stems(text: str) -> list[str]:
    # грубый «стемминг» для тестов: первые 5 букв слова
    return [w[:5] for w in _WORD.findall(normalize(text)) if len(w) > 2]


class FakeEmbedder:
    """Детерминированный «мешок слов» с хешированием: тесты не тянут модели и работают за миллисекунды."""

    dim = 256

    def _vector(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for word in _WORD.findall(text.lower()):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def token_spans(self, text: str) -> list[tuple[int, int]]:
        return [m.span() for m in re.finditer(r"\S+", text)]


class FakeSparse:
    """BM25-подобный sparse-вектор: хеш основ слов."""

    def _vector(self, text: str) -> models.SparseVector:
        counts: dict[int, float] = {}
        for stem in _stems(text):
            idx = int(hashlib.md5(stem.encode()).hexdigest(), 16) % 100_000
            counts[idx] = counts.get(idx, 0.0) + 1.0
        return models.SparseVector(indices=list(counts), values=list(counts.values()))

    def embed_documents(self, texts: Sequence[str]) -> list[models.SparseVector]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> models.SparseVector:
        return self._vector(text)


class FakeReranker:
    """Доля основ слов запроса, которые есть во фрагменте: 0..1, как сигмоида настоящего реранкера."""

    def __init__(self):
        self.calls = 0

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        self.calls += 1
        q = set(_stems(query))
        return [len(q & set(_stems(t))) / len(q) if q else 0.0 for t in texts]


class FakeLLM:
    """Возвращает заранее заданный ответ и запоминает, что ему прислали."""

    def __init__(self, reply: str):
        self.reply = reply
        self.calls: list[list[dict[str, str]]] = []

    async def complete(self, messages, max_tokens=None, temperature=None):
        self.calls.append(messages)
        return Completion(text=self.reply, model="fake", input_tokens=0, output_tokens=0, latency_s=0.0)

    async def aclose(self):
        pass


Reply = str | Exception | list | Callable[[list[dict]], str]


class ScriptedLLM:
    """LLM для тестов графа: отвечает по системному промпту узла.

    replies: {"route": ..., "grade": ..., "rewrite": ..., "answer": ..., "quiz": ..., "summary": ..., "chitchat": ...}
    Значение — строка, исключение (будет брошено), список (по одному на вызов) или функция от messages."""

    KINDS = {
        "маршрутизатор": "route",
        "проверяешь результаты поиска": "grade",
        "переписываешь вопрос": "rewrite",
        "ассистент студента": "answer",
        "составляешь тест": "quiz",
        "Сожми начало диалога": "summary",
        "дружелюбный ассистент": "chitchat",
    }

    def __init__(self, **replies: Reply):
        self.replies = replies
        self.calls: list[tuple[str, list[dict]]] = []
        self.model = "scripted"
        self.reranker = None

    def kinds(self) -> list[str]:
        return [k for k, _ in self.calls]

    def _reply(self, messages: list[dict]) -> str:
        system = messages[0]["content"]
        kind = next(v for k, v in self.KINDS.items() if k in system)
        self.calls.append((kind, messages))
        reply = self.replies.get(kind, f"<нет ответа для {kind}>")
        if isinstance(reply, list):
            reply = reply.pop(0)
        if callable(reply) and not isinstance(reply, Exception):
            reply = reply(messages)
        if isinstance(reply, Exception):
            raise reply
        return reply

    async def complete(self, messages, max_tokens=None, temperature=None) -> Completion:
        return _completion(self._reply(messages))

    async def stream(self, messages, max_tokens=None, temperature=None):
        text = self._reply(messages)
        for token in re.findall(r"\S+\s*", text):
            yield token
        yield _completion(text)

    async def complete_json(self, messages, schema, max_tokens=None):
        text = self._reply(messages)
        return parse_json(text, schema), _completion(text)

    async def aclose(self):
        pass


def _completion(text: str) -> Completion:
    return Completion(text=text, model="scripted", input_tokens=10, output_tokens=5, latency_s=0.0, provider="fake")


class LLMDown(LLMError):
    pass


def make_pdf(pages: Sequence[str]) -> bytes:
    """Строка страницы, начинающаяся с «# », становится заголовком (крупный кегль)."""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        html = "".join(
            f"<h1>{line[2:]}</h1>" if line.startswith("# ") else f"<p>{line}</p>" for line in text.split("\n")
        )
        page.insert_htmlbox(page.rect + (50, 50, -50, -50), html)
    data = doc.tobytes()
    doc.close()
    return data


def make_settings(**overrides: Any) -> Settings:
    base = dict(
        top_k=3,
        rerank_candidates=10,
        relevance_threshold=0.6,
        relevance_floor=0.2,
        max_rewrites=2,
        summary_trigger_tokens=10_000,
        summary_keep_messages=4,
        quiz_questions=3,
        node_timeout=30,
        _env_file=None,
    )
    return Settings(**{**base, **overrides})


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def sparse() -> FakeSparse:
    return FakeSparse()


@pytest.fixture
def reranker() -> FakeReranker:
    return FakeReranker()


@pytest.fixture
def store() -> ChunkStore:
    return ChunkStore(QdrantClient(":memory:"), "test_chunks")


@pytest.fixture
def indexer(embedder, store, sparse) -> Indexer:
    return Indexer(embedder, store, chunk_tokens=40, chunk_overlap=5, sparse=sparse, chunk_min_tokens=3)


@pytest.fixture
def retriever(embedder, store, sparse, reranker) -> Retriever:
    return Retriever(embedder, store, sparse=sparse, reranker=reranker, mode="hybrid", candidates=10)
