"""Поиск по материалам: dense или гибрид (dense + BM25 через RRF), затем по желанию реранкер.

Режимы нужны для оценки: dense → hybrid → hybrid + reranker сравниваются на одном golden set.
"""

import logging
import time

from app.retrieval.embeddings import Embedder
from app.retrieval.reranker import Reranker
from app.retrieval.sparse import SparseEncoder
from app.retrieval.store import ChunkStore, ScoredChunk

log = logging.getLogger(__name__)


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        store: ChunkStore,
        sparse: SparseEncoder | None = None,
        reranker: Reranker | None = None,
        mode: str = "hybrid",
        candidates: int = 30,
    ):
        if mode not in ("dense", "hybrid"):
            raise ValueError(f"неизвестный режим поиска: {mode}")
        if mode == "hybrid" and sparse is None:
            raise ValueError("для гибридного поиска нужен sparse-энкодер")
        self.embedder = embedder
        self.store = store
        self.sparse = sparse
        self.reranker = reranker
        self.mode = mode
        self.candidates = candidates

    def search(self, query: str, top_k: int, course: str | None = None) -> list[ScoredChunk]:
        if not self.store.exists():
            return []
        started = time.perf_counter()
        # Реранкеру нужен запас кандидатов: он поднимает наверх то, что поиск поставил на 15-е место.
        limit = max(self.candidates, top_k) if self.reranker else top_k
        dense = self.embedder.embed_query(query)
        if self.mode == "hybrid":
            hits = self.store.search_hybrid(dense, self.sparse.embed_query(query), limit=limit, course=course)
        else:
            hits = self.store.search_dense(dense, limit=limit, course=course)
        found = time.perf_counter()

        if self.reranker and hits:
            scores = self.reranker.score(query, [_rerank_text(h) for h in hits])
            hits = sorted((h.with_rerank(s) for h, s in zip(hits, scores, strict=True)),
                          key=lambda h: h.rerank_score, reverse=True)
        log.info(
            "retrieve mode=%s rerank=%s candidates=%d search=%.2fs rerank=%.2fs",
            self.mode, bool(self.reranker), len(hits), found - started, time.perf_counter() - found,
        )
        return hits[:top_k]


def _rerank_text(hit: ScoredChunk) -> str:
    return f"{hit.section}\n{hit.text}" if hit.section else hit.text
