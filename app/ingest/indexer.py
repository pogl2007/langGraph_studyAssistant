"""Индексация файла: парсинг → чанки → эмбеддинги → Qdrant.

Всё здесь синхронное и CPU-bound. Из асинхронного кода вызывать через asyncio.to_thread,
иначе парсинг и эмбеддинги заблокируют event loop.
"""

import hashlib
import logging
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from app.ingest.chunking import chunk_pages
from app.ingest.parsers import parse_pdf
from app.retrieval.embeddings import Embedder
from app.retrieval.store import ChunkStore, StoredChunk, point_id

log = logging.getLogger(__name__)

_FILE_NAMESPACE = uuid.UUID("7f1c1a52-3c8e-4d2b-9a57-2f0f5d3c9e11")
_UNSAFE = re.compile(r"[^\w.\- ]+", re.UNICODE)


@dataclass(frozen=True)
class IndexResult:
    file_id: str
    filename: str
    pages: int
    chunks: int
    seconds: float


def safe_filename(name: str) -> str:
    """Оставляет только имя файла без каталогов и опасных символов: «../../etc/passwd» → «passwd»."""
    base = Path(name.replace("\\", "/")).name
    base = _UNSAFE.sub("_", base).strip(" .")
    if not base:
        raise ValueError("пустое имя файла")
    return base[:200]


def make_file_id(course: str, filename: str) -> str:
    # id зависит от курса и имени, а не от содержимого: новая версия того же файла
    # заменяет старую, а не лежит рядом с ней.
    return str(uuid.uuid5(_FILE_NAMESPACE, f"{course}/{filename}"))


class Indexer:
    def __init__(self, embedder: Embedder, store: ChunkStore, chunk_tokens: int, chunk_overlap: int):
        self.embedder = embedder
        self.store = store
        self.chunk_tokens = chunk_tokens
        self.chunk_overlap = chunk_overlap

    def index_pdf(self, filename: str, data: bytes, course: str = "default") -> IndexResult:
        started = time.perf_counter()
        filename = safe_filename(filename)
        file_id = make_file_id(course, filename)

        pages = parse_pdf(data)
        chunks = chunk_pages(pages, self.embedder.token_spans, self.chunk_tokens, self.chunk_overlap)
        file_meta = {
            "file_id": file_id,
            "filename": filename,
            "course": course,
            "file_hash": hashlib.sha256(data).hexdigest(),
            "pages_total": len(pages),
            "chunks_total": len(chunks),
        }
        stored = [
            StoredChunk(
                point_id=point_id(file_id, c.chunk_index),
                text=c.text,
                payload={
                    **file_meta,
                    "page": c.page,
                    "section": c.section,
                    "chunk_index": c.chunk_index,
                    "content_hash": c.content_hash,
                },
            )
            for c in chunks
        ]
        vectors = self.embedder.embed_documents([c.text for c in stored]) if stored else []

        self.store.ensure_collection(self.embedder.dim)
        # id точек детерминированы, поэтому upsert перезаписывает старую версию файла на месте,
        # а потом удаляем хвост, если новая версия короче. Окна «файл пропал из поиска» нет.
        if stored:
            self.store.upsert(stored, vectors)
        self.store.delete_file(file_id, from_chunk_index=len(stored))

        result = IndexResult(file_id, filename, len(pages), len(chunks), time.perf_counter() - started)
        log.info("индексирован %s: %d стр., %d чанков, %.1f с", filename, result.pages, result.chunks, result.seconds)
        return result
