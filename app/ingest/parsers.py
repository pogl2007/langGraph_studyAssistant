"""Извлечение текста из PDF со структурой: заголовки разделов, абзацы, без колонтитулов и обрывков формул.

Лекции чаще всего — слайды, экспортированные в PDF. В них:
- заголовок раздела набран крупнее основного текста («• Закон Кулона» 20 pt против 14 pt)
  и часто лежит в одном блоке с абзацем — поэтому заголовки ищем по строкам, а не по блокам;
- формулы MathType разваливаются на обрывки вроде «Кл 10 6,1 19 e» — для поиска это шум;
- предложение иногда разбито на отдельные строки по словам с одной и той же координатой y;
- имя преподавателя или номер страницы повторяются на каждом слайде.
"""

import re
from collections import Counter
from dataclasses import dataclass, field

import pymupdf

HEADING_SCALE = 1.25  # во сколько раз заголовок крупнее основного текста
MAX_HEADING_CHARS = 150


@dataclass(frozen=True)
class Block:
    text: str
    is_heading: bool = False


@dataclass(frozen=True)
class ParsedPage:
    page: int  # номер страницы с единицы — как в просмотрщике PDF и в цитатах
    text: str
    blocks: tuple[Block, ...] = field(default=())


@dataclass(frozen=True)
class _Row:
    text: str
    size: float  # кегль, которым набрано больше всего символов строки
    bold: bool
    block: int  # номер блока на странице: между блоками — граница абзаца


def parse_pdf(data: bytes) -> list[ParsedPage]:
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        raw = [_page_rows(page) for page in doc]

    body_size = _body_font_size(raw)
    repeated = _repeated_rows(raw)

    pages = []
    for number, rows in enumerate(raw, start=1):
        blocks: list[Block] = []
        paragraph: list[str] = []
        last_block = None

        def flush() -> None:
            # переводы строк внутри абзаца — это вёрстка, а не смысл: склеиваем в одну строку
            text = _SPACE_BEFORE_PUNCT.sub(r"\1", clean_text("\n".join(paragraph)).replace("\n", " "))
            if text:
                blocks.append(Block(text))
            paragraph.clear()

        for row in rows:
            if _norm(row.text) in repeated:
                continue
            if _is_heading(row, body_size):
                flush()
                blocks.append(Block(row.text.lstrip("•▪●-– ").strip(), is_heading=True))
            elif not _is_formula_debris(row.text):
                if last_block is not None and row.block != last_block:
                    flush()
                paragraph.append(row.text)
            last_block = row.block
        flush()
        pages.append(ParsedPage(page=number, text="\n\n".join(b.text for b in blocks), blocks=tuple(blocks)))
    return pages


def _page_rows(page: pymupdf.Page) -> list[_Row]:
    """Строки страницы в порядке чтения."""
    rows: list[_Row] = []
    for block_no, block in enumerate(page.get_text("dict", sort=True)["blocks"]):
        spans = [
            (s["bbox"][1], s["bbox"][0], s["size"], _PRIVATE_USE.sub("", s["text"]), _bold(s))
            for line in block.get("lines", [])
            for s in line["spans"]
        ]
        spans = [s for s in spans if s[3].strip()]
        rows.extend(_Row(*r, block=block_no) for r in _group_rows(spans))
    return rows


def _group_rows(spans: list[tuple]) -> list[tuple[str, float, bool]]:
    """Склеивает спаны одной высоты в строку: «Это» «утверждение» «называется» → одна строка."""
    groups: list[list[tuple]] = []
    for span in sorted(spans, key=lambda s: (s[0], s[1])):
        if groups and abs(groups[-1][0][0] - span[0]) <= span[2] * 0.4:
            groups[-1].append(span)
        else:
            groups.append([span])

    result = []
    for group in groups:
        group.sort(key=lambda s: s[1])
        by_size: Counter[float] = Counter()
        for s in group:
            by_size[round(s[2], 1)] += len(s[3].strip())
        text = " ".join(s[3].strip() for s in group)
        result.append((text, by_size.most_common(1)[0][0], all(s[4] for s in group)))
    return result


def _bold(span: dict) -> bool:
    return bool(span["flags"] & 16) or "bold" in span["font"].lower()


def _body_font_size(raw: list[list[_Row]]) -> float:
    sizes: Counter[float] = Counter()
    for rows in raw:
        for r in rows:
            sizes[round(r.size)] += len(r.text)
    return sizes.most_common(1)[0][0] if sizes else 12.0


def _repeated_rows(raw: list[list[_Row]]) -> set[str]:
    """Строки, которые встречаются на большинстве страниц, — колонтитулы."""
    if len(raw) < 4:
        return set()
    seen: Counter[str] = Counter()
    for rows in raw:
        seen.update({_norm(r.text) for r in rows})
    return {row for row, n in seen.items() if n >= len(raw) * 0.5 and row}


_PRIVATE_USE = re.compile(r"[-]")
_DIGITS = re.compile(r"\d+")
_WORD = re.compile(r"[^\W\d_]{3,}")
_NUMBERED = re.compile(r"^\d+(\.\d+)*\.?\s+\S")


def _is_heading(row: _Row, body_size: float) -> bool:
    if len(row.text) > MAX_HEADING_CHARS or not _WORD.search(row.text):
        return False
    if row.size >= body_size * HEADING_SCALE:
        return True
    # в статьях заголовки того же кегля, но жирные и с номером: «2.1. Rule-based chatbots»
    return row.bold and len(row.text) <= 100 and bool(_NUMBERED.match(row.text))


def _norm(row: str) -> str:
    # «Information 2022, 1, 0 14 of 24» и «… 15 of 24» — один и тот же колонтитул
    return _DIGITS.sub("#", row.strip().lower())


def _is_formula_debris(row: str) -> bool:
    # обрывки формул: «k 9 10 9 (Н м 2 ) Кл 2» — ни одного слова длиннее двух букв
    return not _WORD.search(row)


_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:!?)])")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_SPACES = re.compile(r"[ \t ]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Склеивает переносы слов, схлопывает пробелы. Переводы строк оставляем — по ним режем на абзацы."""
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _MANY_NEWLINES.sub("\n\n", text).strip()
