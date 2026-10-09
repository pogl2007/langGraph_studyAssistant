"""Состояние графа, контекст запуска и схемы структурированного вывода.

Состояние живёт в checkpointer'е между ходами диалога (thread_id). Поля одного хода
(документы, счётчик переписываний, ответ) сбрасывает узел route в начале каждого хода.
"""

from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

Route = Literal["question", "quiz", "chitchat", "out_of_scope", "progress"]
Grade = Literal["relevant", "irrelevant"]


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    summary: str  # сжатая старая часть диалога

    # поля одного хода
    question: str  # что спросил пользователь
    query: str  # поисковый запрос: сначала вопрос, после rewrite — переписанный
    route: Route
    topic: str | None  # тема теста
    follow_up: bool  # «а что насчёт второго?» — без истории не понять
    documents: list[dict[str, Any]]  # ScoredChunk.to_dict()
    grade: Grade
    rewrite_count: int
    answer: str
    citations: list[dict[str, Any]]
    no_answer: bool
    quiz: dict[str, Any] | None  # с правильными ответами — клиенту уходит без них
    quiz_answers: list[int] | None


@dataclass
class AgentContext:
    """Неизменяемые параметры запуска: кто спрашивает и по какому курсу ищем."""

    user_id: str = "anonymous"
    course: str | None = None


class RouteDecision(BaseModel):
    route: Literal["question", "quiz", "chitchat", "out_of_scope"] = Field(
        description="question — вопрос по учебному материалу; quiz — просьба составить тест; "
        "chitchat — приветствие, благодарность, разговор о самом ассистенте; "
        "out_of_scope — просьба, не связанная с учёбой (стихи, погода, код для работы)"
    )
    topic: str | None = Field(default=None, description="тема теста, если route=quiz и тема названа")


class GradeDecision(BaseModel):
    relevant: bool = Field(description="есть ли во фрагментах ответ на вопрос")
    reason: str = Field(description="одно короткое предложение почему")


class QuizQuestion(BaseModel):
    question: str
    options: list[str] = Field(min_length=3, max_length=5)
    correct: int = Field(description="индекс правильного варианта, с нуля")
    explanation: str = Field(description="почему ответ правильный, одним-двумя предложениями")
    source: int = Field(description="номер документа, на котором основан вопрос")
    topic: str = Field(description="узкая тема вопроса, 2–5 слов")


class Quiz(BaseModel):
    topic: str
    questions: list[QuizQuestion]
