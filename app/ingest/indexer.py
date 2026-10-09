"""Индексация файла: парсинг → чанки → dense + sparse векторы → Qdrant.

Всё здесь синхронное и CPU-bound. Из асинхронного кода вызывать через asyncio.to_thread,
иначе парсинг и эмбеддинги заблокируют event loop.
"""

import hashlib
import logging
import re
import time
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path

from app.ingest.chunking import chunk_pages, chunk_structured
from app.ingest.parsers import parse_pdf
from app.retrieval.embeddings import Embedder
from app.retrieval.sparse import SparseEncoder
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
    status: str = "indexed"  # indexed | unchanged


def safe_filename(name: str) -> str:
    """Оставляет только имя файла без каталогов и опасных символов: «../../etc/passwd» → «passwd»."""
    # Имена из macOS и некоторых мессенджеров приходят в NFD: «й» = «и» + отдельный знак бреве.
    # Без нормализации знак считается «опасным символом» и «Действие» превращается в «Деи_ствие».
    name = unicodedata.normalize("NFC", name)
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
    def __init__(
        self,
        embedder: Embedder,
        store: ChunkStore,
        chunk_tokens: int,
        chunk_overlap: int,
        sparse: SparseEncoder | None = None,
        chunking: str = "structured",
        chunk_min_tokens: int = 60,
        files_dir: Path | None = None,
    ):
        self.embedder = embedder
        self.store = store
        self.sparse = sparse
        self.chunk_tokens = chunk_tokens
        self.chunk_overlap = chunk_overlap
        self.chunk_min_tokens = chunk_min_tokens
        self.chunking = chunking
        self.files_dir = files_dir

    def index_pdf(self, filename: str, data: bytes, course: str = "default", force: bool = False) -> IndexResult:
        started = time.perf_counter()
        filename = safe_filename(filename)
        file_id = make_file_id(course, filename)
        file_hash = hashlib.sha256(data).hexdigest()

        existing = self.store.get_file(file_id) if self.store.exists() else None
        if existing and existing.get("file_hash") == file_hash and not force:
            # Тот же файл загрузили ещё раз — ни парсинга, ни эмбеддингов.
            # Но оригинал сохраняем, если его нет (новый volume): иначе ссылки цитат вели бы в 404.
            self._save_original(file_id, data)
            log.info("пропущен %s: содержимое не изменилось", filename)
            return IndexResult(file_id, filename, existing["pages_total"], existing["chunks_total"],
                               time.perf_counter() - started, status="unchanged")

        pages = parse_pdf(data)
        if self.chunking == "structured":
            chunks = chunk_structured(
                pages, self.embedder.token_spans, self.chunk_tokens, self.chunk_overlap, self.chunk_min_tokens
            )
        else:
            chunks = chunk_pages(pages, self.embedder.token_spans, self.chunk_tokens, self.chunk_overlap)

        file_meta = {
            "file_id": file_id,
            "filename": filename,
            "course": course,
            "file_hash": file_hash,
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
        embed_texts = [c.embed_text for c in chunks]
        dense = self.embedder.embed_documents(embed_texts) if chunks else []
        sparse = self.sparse.embed_documents(embed_texts) if self.sparse and chunks else None

        self.store.ensure_collection(self.embedder.dim)
        # id точек детерминированы, поэтому upsert перезаписывает старую версию файла на месте,
        # а потом удаляем хвост, если новая версия короче. Окна «файл пропал из поиска» нет.
        if stored:
            self.store.upsert(stored, dense, sparse)
        self.store.delete_file(file_id, from_chunk_index=len(stored))

        self._save_original(file_id, data, overwrite=True)

        result = IndexResult(file_id, filename, len(pages), len(chunks), time.perf_counter() - started)
        log.info("индексирован %s: %d стр., %d чанков, %.1f с", filename, result.pages, result.chunks, result.seconds)
        return result

    def _save_original(self, file_id: str, data: bytes, overwrite: bool = False) -> None:
        """Оригинал нужен, чтобы цитата открывала PDF на нужной странице."""
        if not self.files_dir:
            return
        path = self.files_dir / f"{file_id}.pdf"
        if overwrite or not path.exists():
            self.files_dir.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

    def delete(self, file_id: str) -> None:
        if self.store.exists():
            self.store.delete_file(file_id)
        if self.files_dir:
            (self.files_dir / f"{file_id}.pdf").unlink(missing_ok=True)
