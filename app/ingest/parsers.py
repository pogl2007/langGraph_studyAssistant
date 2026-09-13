"""Извлечение текста из файлов. Пока только PDF с текстовым слоем; DOCX, Markdown и OCR — позже."""

import re
from dataclasses import dataclass

import pymupdf


@dataclass(frozen=True)
class ParsedPage:
    page: int  # номер страницы с единицы — как в просмотрщике PDF и в цитатах
    text: str


def parse_pdf(data: bytes) -> list[ParsedPage]:
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        return [ParsedPage(page=i + 1, text=clean_text(page.get_text("text"))) for i, page in enumerate(doc)]


_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_SPACES = re.compile(r"[ \t ]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Склеивает переносы слов, схлопывает пробелы. Переводы строк оставляем — по ним потом ищем заголовки."""
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _MANY_NEWLINES.sub("\n\n", text).strip()
