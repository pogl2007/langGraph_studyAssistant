import pytest

from app.ingest.chunking import chunk_pages, chunk_structured
from app.ingest.parsers import Block, ParsedPage, clean_text, parse_pdf

from .conftest import FakeEmbedder, make_pdf

spans = FakeEmbedder().token_spans


def words(n: int, start: int = 0) -> str:
    return " ".join(f"слово{i}" for i in range(start, start + n))


def test_parse_pdf_keeps_page_numbers():
    data = make_pdf(["Первая страница про градиентный спуск", "Вторая страница про регуляризацию"])
    pages = parse_pdf(data)
    assert [p.page for p in pages] == [1, 2]
    assert "градиентный спуск" in pages[0].text
    assert "регуляризацию" in pages[1].text


def test_clean_text_joins_hyphenated_words():
    assert clean_text("обра-\nботка   данных") == "обработка данных"


def test_chunks_respect_size_and_overlap():
    pages = [ParsedPage(1, words(100))]
    chunks = chunk_pages(pages, spans, chunk_tokens=30, overlap=5)

    assert all(len(c.text.split()) <= 30 for c in chunks)
    # соседние чанки перекрываются ровно на 5 слов
    for a, b in zip(chunks, chunks[1:]):
        assert a.text.split()[-5:] == b.text.split()[:5]
    # ничего не потеряли
    covered = set()
    for c in chunks:
        covered.update(c.text.split())
    assert covered == set(words(100).split())


def test_chunks_do_not_cross_pages_and_carry_metadata():
    pages = [ParsedPage(1, words(50)), ParsedPage(2, ""), ParsedPage(3, words(10, start=50))]
    chunks = chunk_pages(pages, spans, chunk_tokens=30, overlap=5)

    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert {c.page for c in chunks} == {1, 3}  # пустая страница не даёт чанков
    assert chunks[-1].text == words(10, start=50)
    assert all(len(c.content_hash) == 64 for c in chunks)
    assert chunks[0].content_hash != chunks[1].content_hash


def test_short_page_is_single_chunk():
    chunks = chunk_pages([ParsedPage(7, "короткий текст")], spans, chunk_tokens=30, overlap=5)
    assert len(chunks) == 1
    assert chunks[0].page == 7


def test_overlap_must_be_smaller_than_chunk():
    with pytest.raises(ValueError):
        chunk_pages([ParsedPage(1, "a b c")], spans, chunk_tokens=10, overlap=10)


def test_subword_tokens_are_snapped_to_word_boundaries():
    text = "обработка естественного языка"

    def subword_spans(t):  # имитируем токенизатор, режущий слова на куски по 3 символа
        return [(i, min(i + 3, len(t))) for i in range(0, len(t)) if i % 3 == 0 and not t[i].isspace()]

    chunks = chunk_pages([ParsedPage(1, text)], subword_spans, chunk_tokens=2, overlap=0)
    for c in chunks:
        for w in c.text.split():
            assert w in text.split()


def test_structured_parser_detects_headings_and_drops_formula_debris():
    data = make_pdf(["# Закон Кулона\nСила обратно пропорциональна квадрату расстояния.\nk 9 10 9\n# Точечный заряд\nТело малых размеров."])
    [page] = parse_pdf(data)
    headings = [b.text for b in page.blocks if b.is_heading]
    assert headings == ["Закон Кулона", "Точечный заряд"]
    assert "k 9 10 9" not in page.text  # обрывок формулы


def test_structured_chunks_follow_sections_and_carry_title_to_next_page():
    pages = [
        ParsedPage(1, "", (Block("Закон Кулона", True), Block(words(20)))),
        ParsedPage(2, "", (Block(words(20, start=100)),)),  # продолжение раздела без заголовка
        ParsedPage(3, "", (Block("Теорема Гаусса", True), Block(words(20, start=200)))),
    ]
    chunks = chunk_structured(pages, spans, chunk_tokens=30, overlap=5, min_tokens=3)
    assert [(c.page, c.section) for c in chunks] == [(1, "Закон Кулона"), (2, "Закон Кулона"), (3, "Теорема Гаусса")]
    assert chunks[0].embed_text.startswith("Закон Кулона\n")


def test_small_sections_are_merged_on_the_same_page():
    pages = [ParsedPage(1, "", (Block("A", True), Block("коротко"), Block("B", True), Block(words(30))))]
    chunks = chunk_structured(pages, spans, chunk_tokens=50, overlap=5, min_tokens=5)
    assert len(chunks) == 1
    assert chunks[0].section == "A; B"


def test_large_section_is_split_recursively_with_overlap():
    text = "\n\n".join(" ".join(f"п{p}с{i}." for i in range(20)) for p in range(4))
    pages = [ParsedPage(1, "", (Block("Большой раздел", True), Block(text)))]
    chunks = chunk_structured(pages, spans, chunk_tokens=30, overlap=8, min_tokens=3)
    assert len(chunks) > 2
    assert all(len(c.text.split()) <= 30 for c in chunks)
    assert all(c.section == "Большой раздел" for c in chunks)
