"""Работа с Qdrant: коллекция с dense- и sparse-векторами, запись чанков, удаление файла, поиск.

Qdrant — единственный источник правды о проиндексированных файлах: сведения о файле лежат
в payload каждого чанка, а список файлов строится по чанкам с chunk_index == 0.
"""

import uuid
from dataclasses import asdict, dataclass, replace
from typing import Any

from qdrant_client import QdrantClient, models

DENSE = "dense"
SPARSE = "bm25"


@dataclass(frozen=True)
class StoredChunk:
    point_id: str
    text: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class ScoredChunk:
    text: str
    score: float  # скор поиска: косинус (dense) или RRF (hybrid)
    file_id: str
    filename: str
    page: int
    chunk_index: int
    section: str | None
    rerank_score: float | None = None  # скор cross-encoder, если реранкер включён

    @property
    def relevance(self) -> float:
        """Скор, по которому решаем «нашлось или нет»: реранкер, если он был, иначе скор поиска."""
        return self.rerank_score if self.rerank_score is not None else self.score

    def with_rerank(self, score: float) -> "ScoredChunk":
        return replace(self, rerank_score=score)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScoredChunk":
        return cls(**data)


def point_id(file_id: str, chunk_index: int) -> str:
    # Детерминированный id: повторная индексация того же файла перезаписывает точки, а не плодит дубли.
    return str(uuid.uuid5(uuid.UUID(file_id), str(chunk_index)))


class ChunkStore:
    def __init__(self, client: QdrantClient, collection: str):
        self.client = client
        self.collection = collection

    def exists(self) -> bool:
        return self.client.collection_exists(self.collection)

    def ensure_collection(self, dim: int) -> None:
        if self.exists():
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
            # IDF считает Qdrant по всей коллекции: новые лекции сразу меняют вес редких терминов
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        for field, schema in [
            ("file_id", models.PayloadSchemaType.KEYWORD),
            ("course", models.PayloadSchemaType.KEYWORD),
            ("chunk_index", models.PayloadSchemaType.INTEGER),
        ]:
            self.client.create_payload_index(self.collection, field_name=field, field_schema=schema)

    def upsert(
        self,
        chunks: list[StoredChunk],
        dense: list[list[float]],
        sparse: list[models.SparseVector] | None = None,
    ) -> None:
        sparse = sparse or [None] * len(chunks)
        points = [
            models.PointStruct(
                id=c.point_id,
                vector={DENSE: d, SPARSE: s} if s is not None else {DENSE: d},
                payload={"text": c.text, **c.payload},
            )
            for c, d, s in zip(chunks, dense, sparse, strict=True)
        ]
        self.client.upsert(self.collection, points=points, wait=True)

    def delete_file(self, file_id: str, from_chunk_index: int = 0) -> None:
        conditions = [models.FieldCondition(key="file_id", match=models.MatchValue(value=file_id))]
        if from_chunk_index > 0:
            conditions.append(models.FieldCondition(key="chunk_index", range=models.Range(gte=from_chunk_index)))
        self.client.delete(
            self.collection,
            points_selector=models.FilterSelector(filter=models.Filter(must=conditions)),
            wait=True,
        )

    def list_files(self, course: str | None = None) -> list[dict[str, Any]]:
        if not self.exists():
            return []
        conditions = [models.FieldCondition(key="chunk_index", match=models.MatchValue(value=0))]
        if course:
            conditions.append(models.FieldCondition(key="course", match=models.MatchValue(value=course)))
        files = [
            {k: v for k, v in payload.items() if k not in ("text", "section", "page", "chunk_index", "content_hash")}
            for payload in self._scroll(models.Filter(must=conditions), with_payload=True)
        ]
        return sorted(files, key=lambda f: f["filename"])

    def get_file(self, file_id: str) -> dict[str, Any] | None:
        return next((f for f in self.list_files() if f["file_id"] == file_id), None)

    def list_sections(self, course: str | None = None) -> list[dict[str, Any]]:
        """Разделы материалов в порядке следования: для выбора темы теста и подсказок «ближайшие темы»."""
        if not self.exists():
            return []
        flt = _match("course", course) if course else None
        seen: dict[tuple[str, str], dict[str, Any]] = {}
        for p in self._scroll(flt, with_payload=["section", "filename", "page", "file_id", "chunk_index"]):
            if p.get("section") and (p["file_id"], p["section"]) not in seen:
                seen[(p["file_id"], p["section"])] = p
        return sorted(seen.values(), key=lambda p: (p["filename"], p["chunk_index"]))

    def search_dense(self, vector: list[float], limit: int, course: str | None = None) -> list[ScoredChunk]:
        flt = _match("course", course) if course else None
        response = self.client.query_points(
            self.collection, query=vector, using=DENSE, limit=limit, query_filter=flt, with_payload=True
        )
        return [_to_scored(p) for p in response.points]

    def search_sparse(self, vector: models.SparseVector, limit: int, course: str | None = None) -> list[ScoredChunk]:
        flt = _match("course", course) if course else None
        response = self.client.query_points(
            self.collection, query=vector, using=SPARSE, limit=limit, query_filter=flt, with_payload=True
        )
        return [_to_scored(p) for p in response.points]

    def search_hybrid(
        self,
        dense: list[float],
        sparse: models.SparseVector,
        limit: int,
        course: str | None = None,
        prefetch_limit: int | None = None,
    ) -> list[ScoredChunk]:
        """Dense и BM25 ищут независимо, Qdrant сливает списки через RRF (Query API, один запрос)."""
        flt = _match("course", course) if course else None
        prefetch_limit = prefetch_limit or max(limit, 30)
        response = self.client.query_points(
            self.collection,
            prefetch=[
                models.Prefetch(query=dense, using=DENSE, limit=prefetch_limit, filter=flt),
                models.Prefetch(query=sparse, using=SPARSE, limit=prefetch_limit, filter=flt),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return [_to_scored(p) for p in response.points]

    def _scroll(self, flt: models.Filter | None, with_payload) -> list[dict[str, Any]]:
        result, offset = [], None
        while True:
            points, offset = self.client.scroll(
                self.collection, scroll_filter=flt, limit=256, offset=offset, with_payload=with_payload, with_vectors=False
            )
            result.extend(p.payload or {} for p in points)
            if offset is None:
                return result


def _match(key: str, value: Any) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key=key, match=models.MatchValue(value=value))])


def _to_scored(point: models.ScoredPoint) -> ScoredChunk:
    p = point.payload or {}
    return ScoredChunk(
        text=p["text"],
        score=point.score,
        file_id=p["file_id"],
        filename=p["filename"],
        page=p["page"],
        chunk_index=p["chunk_index"],
        section=p.get("section"),
    )
