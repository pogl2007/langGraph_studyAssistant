"""Работа с Qdrant: коллекция, запись чанков, удаление файла, dense-поиск.

Qdrant — единственный источник правды о проиндексированных файлах: сведения о файле лежат
в payload каждого чанка, а список файлов строится по чанкам с chunk_index == 0.
"""

import uuid
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient, models

DENSE = "dense"


@dataclass(frozen=True)
class StoredChunk:
    point_id: str
    text: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class ScoredChunk:
    text: str
    score: float
    file_id: str
    filename: str
    page: int
    chunk_index: int
    section: str | None


def point_id(file_id: str, chunk_index: int) -> str:
    # Детерминированный id: повторная индексация того же файла перезаписывает точки, а не плодит дубли.
    return str(uuid.uuid5(uuid.UUID(file_id), str(chunk_index)))


class ChunkStore:
    def __init__(self, client: QdrantClient, collection: str):
        self.client = client
        self.collection = collection

    def ensure_collection(self, dim: int) -> None:
        if self.client.collection_exists(self.collection):
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
        )
        for field, schema in [
            ("file_id", models.PayloadSchemaType.KEYWORD),
            ("course", models.PayloadSchemaType.KEYWORD),
            ("chunk_index", models.PayloadSchemaType.INTEGER),
        ]:
            self.client.create_payload_index(self.collection, field_name=field, field_schema=schema)

    def upsert(self, chunks: list[StoredChunk], vectors: list[list[float]]) -> None:
        points = [
            models.PointStruct(id=c.point_id, vector={DENSE: v}, payload={"text": c.text, **c.payload})
            for c, v in zip(chunks, vectors, strict=True)
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

    def list_files(self) -> list[dict[str, Any]]:
        files, offset = [], None
        while True:
            points, offset = self.client.scroll(
                self.collection,
                scroll_filter=_match("chunk_index", 0),
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            files.extend({k: v for k, v in p.payload.items() if k != "text"} for p in points)
            if offset is None:
                return sorted(files, key=lambda f: f["filename"])

    def search_dense(self, vector: list[float], limit: int, course: str | None = None) -> list[ScoredChunk]:
        response = self.client.query_points(
            self.collection,
            query=vector,
            using=DENSE,
            limit=limit,
            query_filter=_match("course", course) if course else None,
            with_payload=True,
        )
        return [_to_scored(p) for p in response.points]


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
