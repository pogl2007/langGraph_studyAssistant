"""Простой чанкинг для MVP: окно фиксированной длины в токенах с перекрытием, внутри одной страницы.

Чанк не пересекает границу страницы — тогда у каждого чанка ровно одна страница и цитата
[файл, стр. N] всегда точная. Чанкинг по структуре документа появится на этапе 3.
"""

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from app.ingest.parsers import ParsedPage

# Функция, которая возвращает для текста границы токенов в символах: [(start, end), ...].
# Считаем токены тем же токенизатором, что у эмбеддинг-модели, иначе «500 токенов» ничего не значат.
TokenSpans = Callable[[str], Sequence[tuple[int, int]]]


@dataclass(frozen=True)
class Chunk:
    text: str
    page: int
    chunk_index: int
    section: str | None = None

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def chunk_pages(
    pages: Sequence[ParsedPage],
    token_spans: TokenSpans,
    chunk_tokens: int = 500,
    overlap: int = 60,
) -> list[Chunk]:
    if not 0 <= overlap < chunk_tokens:
        raise ValueError("перекрытие должно быть меньше размера чанка")

    chunks: list[Chunk] = []
    for page in pages:
        for text in _split_page(page.text, token_spans, chunk_tokens, overlap):
            chunks.append(Chunk(text=text, page=page.page, chunk_index=len(chunks)))
    return chunks


def _split_page(text: str, token_spans: TokenSpans, chunk_tokens: int, overlap: int) -> list[str]:
    spans = [s for s in token_spans(text) if s[1] > s[0]]
    if not spans:
        return []

    pieces: list[str] = []
    step = chunk_tokens - overlap
    start = 0
    while True:
        window = spans[start : start + chunk_tokens]
        begin, end = _snap_to_words(text, window[0][0], window[-1][1])
        piece = text[begin:end].strip()
        if piece and (not pieces or piece != pieces[-1]):
            pieces.append(piece)
        if start + chunk_tokens >= len(spans):
            break
        start += step
    return pieces


def _snap_to_words(text: str, begin: int, end: int) -> tuple[int, int]:
    """Токены бывают кусками слов; расширяем окно до границ слов, чтобы не резать «обраб|отка»."""
    while begin > 0 and not text[begin - 1].isspace():
        begin -= 1
    while end < len(text) and not text[end].isspace():
        end += 1
    return begin, end
