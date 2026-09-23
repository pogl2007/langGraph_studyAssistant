"""Превращает ссылки модели [1], [2, 3] в цитаты [файл, стр. N].

Номер, которого нет среди найденных документов, выбрасывается: модель не может сослаться
на источник, которого ей не давали.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from app.answer.prompts import NO_ANSWER
from app.retrieval.store import ScoredChunk

_REF = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")
_DUP = re.compile(r"(\[[^\[\]]+, стр\. \d+\])(?:\s*\1)+")
# Модели иногда дописывают свой список источников: «Ссылки:» и строки «[1] файл.pdf, стр. 3».
# Список под ответом собираем мы, поэтому такие строки выбрасываем.
_LIST_HEADER = re.compile(r"^\s*(\*\*)?(ссылки|источники)(\*\*)?\s*:?\s*(\*\*)?\s*$", re.IGNORECASE)
_LIST_ENTRY = re.compile(r"^\s*(?:\[\d+\]\s*[^\[\]]*?стр\.\s*\d+\s*[;.,]?\s*)+$")


@dataclass(frozen=True)
class Citation:
    filename: str
    page: int
    file_id: str
    snippet: str

    @property
    def label(self) -> str:
        return f"[{self.filename}, стр. {self.page}]"


def render_citations(answer: str, chunks: Sequence[ScoredChunk]) -> tuple[str, list[Citation]]:
    cited: dict[tuple[str, int], Citation] = {}

    def replace(match: re.Match) -> str:
        labels = []
        for raw in re.split(r"\s*[,;]\s*", match.group(1)):
            n = int(raw)
            if not 1 <= n <= len(chunks):
                continue
            c = chunks[n - 1]
            citation = cited.setdefault((c.file_id, c.page), Citation(c.filename, c.page, c.file_id, c.text[:300]))
            if citation.label not in labels:
                labels.append(citation.label)
        return "".join(labels)

    answer = "\n".join(
        line for line in answer.split("\n") if not (_LIST_HEADER.match(line) or _LIST_ENTRY.match(line))
    )
    text = _REF.sub(replace, answer)
    text = _DUP.sub(r"\1", text)
    text = re.sub(r"[ \t]+([.,;:])", r"\1", text).strip()
    return text, list(cited.values())


def is_refusal(answer: str) -> bool:
    normalized = answer.strip().strip("«»\"' ").lower()
    return normalized.startswith(NO_ANSWER.lower().rstrip("."))
