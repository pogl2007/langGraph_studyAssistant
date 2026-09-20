"""Поиск по материалам. В MVP — только dense; гибрид и реранкер добавятся на этапе 3."""

from app.retrieval.embeddings import Embedder
from app.retrieval.store import ChunkStore, ScoredChunk


class Retriever:
    def __init__(self, embedder: Embedder, store: ChunkStore):
        self.embedder = embedder
        self.store = store

    def search(self, query: str, top_k: int, course: str | None = None) -> list[ScoredChunk]:
        if not self.store.client.collection_exists(self.store.collection):
            return []
        return self.store.search_dense(self.embedder.embed_query(query), limit=top_k, course=course)
