"""Линейный RAG: поиск → генерация → цитаты. Базовая версия, от которой считаем все улучшения.

На этапе 4 эта логика переедет в граф LangGraph, а пайплайн останется точкой отсчёта в оценке.
"""

import asyncio
from dataclasses import dataclass, field

from app.answer.citations import Citation, is_refusal, render_citations
from app.answer.prompts import NO_ANSWER, answer_messages
from app.llm.client import Completion, LLMClient
from app.retrieval.search import Retriever
from app.retrieval.store import ScoredChunk


@dataclass
class Answer:
    text: str
    no_answer: bool
    citations: list[Citation] = field(default_factory=list)
    chunks: list[ScoredChunk] = field(default_factory=list)
    completion: Completion | None = None


class RagPipeline:
    def __init__(self, retriever: Retriever, llm: LLMClient, top_k: int, min_score: float = 0.0):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.min_score = min_score

    async def answer(self, question: str, course: str | None = None) -> Answer:
        # Эмбеддинг запроса и поиск — синхронные и CPU-bound, уводим их из event loop.
        chunks = await asyncio.to_thread(self.retriever.search, question, self.top_k, course)
        if not chunks or chunks[0].score < self.min_score:
            return Answer(text=NO_ANSWER, no_answer=True, chunks=chunks)

        completion = await self.llm.complete(answer_messages(question, chunks))
        if is_refusal(completion.text):
            return Answer(text=NO_ANSWER, no_answer=True, chunks=chunks, completion=completion)

        text, citations = render_citations(completion.text, chunks)
        return Answer(text=text, no_answer=False, citations=citations, chunks=chunks, completion=completion)
