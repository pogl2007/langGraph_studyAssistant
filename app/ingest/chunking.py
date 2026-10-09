"""Чанкинг. Две стратегии:

- `chunk_pages` — простое окно фиксированной длины внутри страницы (MVP, точка отсчёта в оценке);
- `chunk_structured` — по разделам документа: заголовок задаёт границу, большой раздел режется
  рекурсивно (абзацы → строки → предложения → слова) с перекрытием, мелкие соседние разделы
  одной страницы склеиваются.

В обоих случаях чанк не пересекает границу страницы — у каждого чанка ровно одна страница
и цитата [файл, стр. N] всегда точная. Раздел, начатый на одной странице и продолженный на
следующей, сохраняет заголовок.
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

    @property
    def embed_text(self) -> str:
        """Текст для эмбеддинга: заголовок раздела + фрагмент.

        Абзац «Это утверждение называется законом Кулона» сам по себе не говорит, о чём раздел;
        с заголовком вектор и BM25 находят его по названию темы."""
        return f"{self.section}\n{self.text}" if self.section else self.text


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


def chunk_structured(
    pages: Sequence[ParsedPage],
    token_spans: TokenSpans,
    chunk_tokens: int = 500,
    overlap: int = 60,
    min_tokens: int = 60,
) -> list[Chunk]:
    if not 0 <= overlap < chunk_tokens:
        raise ValueError("перекрытие должно быть меньше размера чанка")

    def count(text: str) -> int:
        return len(token_spans(text))

    chunks: list[Chunk] = []
    section: str | None = None
    for page in pages:
        segments: list[tuple[str | None, list[str]]] = []
        for block in page.blocks or ():
            if block.is_heading:
                section = block.text
                segments.append((section, []))
            else:
                if not segments or segments[-1][0] != section:
                    segments.append((section, []))
                segments[-1][1].append(block.text)
        if not page.blocks and page.text.strip():  # страница без разметки — один сегмент
            segments.append((section, [page.text]))

        merged = _merge_small([(s, "\n\n".join(p)) for s, p in segments if p], count, min_tokens)
        for title, text in merged:
            for piece in _split_recursive(text, count, chunk_tokens, overlap):
                chunks.append(Chunk(text=piece, page=page.page, chunk_index=len(chunks), section=title))
    return chunks


def _merge_small(segments: list[tuple[str | None, str]], count, min_tokens: int) -> list[tuple[str | None, str]]:
    """Раздел в пару строк — плохой чанк: мало контекста. Приклеиваем его к следующему на той же странице."""
    result: list[tuple[str | None, str]] = []
    carry: tuple[str | None, str] | None = None
    for title, text in segments:
        if carry:
            title, text = _join_titles(carry[0], title), f"{carry[1]}\n\n{text}"
            carry = None
        if count(text) < min_tokens:
            carry = (title, text)
        else:
            result.append((title, text))
    if carry:
        if result:  # последний раздел страницы мелкий — к предыдущему
            prev_title, prev_text = result.pop()
            result.append((_join_titles(prev_title, carry[0]), f"{prev_text}\n\n{carry[1]}"))
        else:
            result.append(carry)
    return result


def _join_titles(a: str | None, b: str | None) -> str | None:
    if not a or a == b:
        return b
    if not b:
        return a
    return f"{a}; {b}"


_SEPARATORS = ("\n\n", "\n", ". ", " ")


def _split_recursive(text: str, count, max_tokens: int, overlap: int, seps=_SEPARATORS) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if count(text) <= max_tokens or not seps:
        return [text]
    sep, rest = seps[0], seps[1:]
    parts = [p.strip() for p in text.split(sep) if p.strip()]
    if len(parts) == 1:
        return _split_recursive(text, count, max_tokens, overlap, rest)

    units: list[str] = []
    for part in parts:
        units.extend(_split_recursive(part, count, max_tokens, overlap, rest) if count(part) > max_tokens else [part])

    pieces: list[str] = []
    window: list[tuple[str, int]] = []
    for unit in units:
        size = count(unit)
        if window and sum(n for _, n in window) + size > max_tokens:
            pieces.append(sep.join(u for u, _ in window))
            # перекрытие: хвост предыдущего окна, сколько влезает в overlap токенов
            carry: list[tuple[str, int]] = []
            for prev in reversed(window):
                if sum(n for _, n in carry) + prev[1] > overlap:
                    break
                carry.insert(0, prev)
            window = carry
            while window and sum(n for _, n in window) + size > max_tokens:
                window.pop(0)
        window.append((unit, size))
    if window:
        pieces.append(sep.join(u for u, _ in window))
    return pieces


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
