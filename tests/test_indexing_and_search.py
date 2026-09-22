import pytest

from app.ingest.indexer import make_file_id, safe_filename
from app.retrieval.search import Retriever

from .conftest import make_pdf

LECTURE = [
    "Градиентный спуск обновляет веса модели в направлении антиградиента функции потерь.",
    "Регуляризация L2 добавляет к функции потерь штраф за большие веса и снижает переобучение.",
    "Свёрточные сети применяют фильтры к изображению и выделяют локальные признаки.",
]


def test_index_and_find_right_page(indexer, embedder, store):
    result = indexer.index_pdf("lecture1.pdf", make_pdf(LECTURE), course="ml")
    assert (result.pages, result.chunks) == (3, 3)

    hits = Retriever(embedder, store).search("что делает регуляризация L2", top_k=3, course="ml")
    assert hits[0].page == 2
    assert hits[0].filename == "lecture1.pdf"
    assert "штраф" in hits[0].text


def test_list_files_has_metadata(indexer, store):
    indexer.index_pdf("lecture1.pdf", make_pdf(LECTURE), course="ml")
    [f] = store.list_files()
    assert f["filename"] == "lecture1.pdf"
    assert f["course"] == "ml"
    assert f["pages_total"] == 3
    assert f["chunks_total"] == 3
    assert f["file_id"] == make_file_id("ml", "lecture1.pdf")
    assert len(f["file_hash"]) == 64


def test_course_filter(indexer, embedder, store):
    indexer.index_pdf("ml.pdf", make_pdf(LECTURE), course="ml")
    indexer.index_pdf("history.pdf", make_pdf(["Реформы Петра Первого изменили армию."]), course="history")

    hits = Retriever(embedder, store).search("реформы армии", top_k=5, course="ml")
    assert {h.filename for h in hits} == {"ml.pdf"}


def test_reindex_shorter_version_removes_stale_chunks(indexer, store):
    indexer.index_pdf("lecture1.pdf", make_pdf(LECTURE), course="ml")
    indexer.index_pdf("lecture1.pdf", make_pdf(LECTURE[:1]), course="ml")

    points, _ = store.client.scroll(store.collection, limit=100)
    assert len(points) == 1
    assert store.list_files()[0]["chunks_total"] == 1


def test_delete_file(indexer, embedder, store):
    r = indexer.index_pdf("lecture1.pdf", make_pdf(LECTURE), course="ml")
    store.delete_file(r.file_id)
    assert store.list_files() == []
    assert Retriever(embedder, store).search("градиентный спуск", top_k=3) == []


def test_search_without_collection_returns_nothing(embedder, store):
    assert Retriever(embedder, store).search("что угодно", top_k=3) == []


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("../../etc/passwd", "passwd"),
        ("..\\..\\windows\\system.ini", "system.ini"),
        ("Лекция 1: введение.pdf", "Лекция 1_ введение.pdf"),
        ("/abs/path/notes.pdf", "notes.pdf"),
    ],
)
def test_safe_filename(raw, expected):
    assert safe_filename(raw) == expected


def test_safe_filename_rejects_empty():
    with pytest.raises(ValueError):
        safe_filename("../..")
