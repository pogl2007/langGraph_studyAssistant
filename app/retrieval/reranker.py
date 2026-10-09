"""Реранкер: cross-encoder BAAI/bge-reranker-v2-m3.

Bi-encoder (bge-m3) кодирует вопрос и фрагмент по отдельности — быстро, но грубо.
Cross-encoder читает пару «вопрос + фрагмент» целиком и оценивает, отвечает ли фрагмент
на вопрос. Это дорого, поэтому он пересортировывает только top-N кандидатов поиска.

Скор после сигмоиды лежит в [0, 1] и сравним между запросами — на нём держится порог
«в материалах этого нет», в отличие от косинуса, где нерелевантный вопрос получал 0,40
при релевантном 0,49.
"""

from collections.abc import Sequence
from typing import Protocol


class Reranker(Protocol):
    def score(self, query: str, texts: Sequence[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str = "cpu", max_length: int = 512, batch_size: int = 16):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = CrossEncoder(model_name, device=device, max_length=max_length)

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        scores = self.model.predict([(query, t) for t in texts], batch_size=self.batch_size, show_progress_bar=False)
        return [float(s) for s in scores]
