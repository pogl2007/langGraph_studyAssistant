"""Sparse-векторы BM25 через fastembed (модель Qdrant/bm25) для гибридного поиска.

IDF считает сам Qdrant (modifier=IDF в коллекции), fastembed даёт только TF-часть.
Поэтому при добавлении документов статистика корпуса обновляется без переиндексации.
"""

from collections.abc import Sequence
from typing import Protocol

from qdrant_client import models


def normalize(text: str) -> str:
    # «напряжённость» в лекции и «напряженность» в запросе — разные токены для BM25:
    # стеммер про «ё» не знает. Без этой строки точное совпадение термина теряется.
    return text.lower().replace("ё", "е")


class SparseEncoder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[models.SparseVector]: ...

    def embed_query(self, text: str) -> models.SparseVector: ...


class Bm25Encoder:
    def __init__(self, language: str = "russian"):
        from fastembed import SparseTextEmbedding

        self.model = SparseTextEmbedding("Qdrant/bm25", language=language)

    def embed_documents(self, texts: Sequence[str]) -> list[models.SparseVector]:
        return [_to_qdrant(e) for e in self.model.embed([normalize(t) for t in texts])]

    def embed_query(self, text: str) -> models.SparseVector:
        return _to_qdrant(next(iter(self.model.query_embed(normalize(text)))))


def _to_qdrant(embedding) -> models.SparseVector:
    return models.SparseVector(
        indices=[int(i) for i in embedding.indices],
        values=[float(v) for v in embedding.values],
    )
