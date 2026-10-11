"""Сборка компонентов из настроек. Одно место, где решается, какие реализации подставить."""

from functools import lru_cache

from qdrant_client import QdrantClient

from app.answer.pipeline import RagPipeline
from app.config import Settings, get_settings
from app.graph.builder import build_graph
from app.graph.nodes import AgentNodes
from app.ingest.indexer import Indexer
from app.llm.client import LLMClient
from app.retrieval.embeddings import SentenceTransformerEmbedder
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.search import Retriever
from app.retrieval.sparse import Bm25Encoder
from app.retrieval.store import ChunkStore


@lru_cache
def get_embedder() -> SentenceTransformerEmbedder:
    s = get_settings()
    return SentenceTransformerEmbedder(s.embedding_model, s.embedding_device, s.embedding_batch_size)


@lru_cache
def get_sparse() -> Bm25Encoder:
    return Bm25Encoder(get_settings().sparse_language)


@lru_cache
def get_reranker() -> CrossEncoderReranker | None:
    s = get_settings()
    if not s.rerank_enabled:
        return None
    return CrossEncoderReranker(s.reranker_model, s.embedding_device, s.rerank_max_length)


@lru_cache
def get_store() -> ChunkStore:
    s = get_settings()
    return ChunkStore(QdrantClient(url=s.qdrant_url, timeout=30), s.qdrant_collection)


def get_indexer() -> Indexer:
    s = get_settings()
    return Indexer(
        get_embedder(),
        get_store(),
        s.chunk_tokens,
        s.chunk_overlap,
        sparse=get_sparse(),
        chunking=s.chunking,
        chunk_min_tokens=s.chunk_min_tokens,
        files_dir=s.files_dir,
    )


def get_retriever(mode: str | None = None, rerank: bool | None = None) -> Retriever:
    s = get_settings()
    mode = mode or s.retrieval_mode
    use_rerank = s.rerank_enabled if rerank is None else rerank
    return Retriever(
        get_embedder(),
        get_store(),
        sparse=get_sparse() if mode == "hybrid" else None,
        reranker=get_reranker() if use_rerank else None,
        mode=mode,
        candidates=s.rerank_candidates,
    )


def warm_up() -> None:
    """Загрузить модели заранее, чтобы первый запрос пользователя не ждал минуту."""
    get_embedder()
    get_sparse()
    get_reranker()


def build_images(settings: Settings | None = None):
    """Генератор картинок есть только у GigaChat; без его ключа узел illustrate честно скажет, что не умеет."""
    s = settings or get_settings()
    if not s.gigachat_credentials.get_secret_value():
        return None
    from app.llm.images import GigaChatImageGenerator

    return GigaChatImageGenerator.from_settings(s)


def build_agent(llm: LLMClient, checkpointer=None, store=None, settings: Settings | None = None, images=None):
    s = settings or get_settings()
    nodes = AgentNodes(get_retriever(), llm, s, list_sections=get_store().list_sections, images=images)
    return build_graph(nodes, checkpointer=checkpointer, store=store, node_timeout=s.node_timeout)


def build_pipeline(settings: Settings | None = None) -> RagPipeline:
    # LLM-клиент не кэшируем: асинхронный HTTP-клиент привязан к event loop, в котором создан.
    s = settings or get_settings()
    return RagPipeline(get_retriever(), LLMClient.from_settings(s), top_k=s.top_k, min_score=s.min_score)
