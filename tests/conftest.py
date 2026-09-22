import hashlib
import math
import re
from collections.abc import Sequence

import pymupdf
import pytest
from qdrant_client import QdrantClient

from app.ingest.indexer import Indexer
from app.llm.client import Completion
from app.retrieval.store import ChunkStore

_WORD = re.compile(r"\w+", re.UNICODE)


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


class FakeLLM:
    """Возвращает заранее заданный ответ и запоминает, что ему прислали."""

    def __init__(self, reply: str):
        self.reply = reply
        self.calls: list[list[dict[str, str]]] = []

    async def complete(self, messages):
        self.calls.append(messages)
        return Completion(text=self.reply, model="fake", input_tokens=0, output_tokens=0, latency_s=0.0)

    async def aclose(self):
        pass


def make_pdf(pages: Sequence[str]) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_htmlbox(page.rect + (50, 50, -50, -50), text.replace("\n", "<br>"))
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def store() -> ChunkStore:
    return ChunkStore(QdrantClient(":memory:"), "test_chunks")


@pytest.fixture
def indexer(embedder, store) -> Indexer:
    return Indexer(embedder, store, chunk_tokens=40, chunk_overlap=5)
