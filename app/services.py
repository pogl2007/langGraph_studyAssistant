"""Сборка компонентов из настроек. Одно место, где решается, какие реализации подставить."""

from functools import lru_cache

from qdrant_client import QdrantClient

from app.answer.pipeline import RagPipeline
from app.config import Settings, get_settings
from app.ingest.indexer import Indexer
from app.llm.client import LLMClient
from app.retrieval.embeddings import SentenceTransformerEmbedder
from app.retrieval.search import Retriever
from app.retrieval.store import ChunkStore


@lru_cache
def get_embedder() -> SentenceTransformerEmbedder:
    s = get_settings()
    return SentenceTransformerEmbedder(s.embedding_model, s.embedding_device, s.embedding_batch_size)


@lru_cache
def get_store() -> ChunkStore:
    s = get_settings()
    return ChunkStore(QdrantClient(url=s.qdrant_url, timeout=30), s.qdrant_collection)


def get_indexer() -> Indexer:
    s = get_settings()
    return Indexer(get_embedder(), get_store(), s.chunk_tokens, s.chunk_overlap)


def get_retriever() -> Retriever:
    return Retriever(get_embedder(), get_store())


def build_pipeline(settings: Settings | None = None) -> RagPipeline:
    # LLM-клиент не кэшируем: асинхронный HTTP-клиент привязан к event loop, в котором создан.
    s = settings or get_settings()
    return RagPipeline(get_retriever(), LLMClient.from_settings(s), top_k=s.top_k, min_score=s.min_score)
