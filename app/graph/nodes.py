"""Узлы графа.

Где граф, а где простой пайплайн: последовательность «найти → ответить» известна заранее, и там
нет никакой автономии. Граф нужен там, где следующий шаг зависит от результата предыдущего:
- что за запрос (вопрос, тест, болтовня, не по теме) — route;
- хватает ли найденного — grade; если нет, переписать запрос и искать снова, но не больше N раз;
- модель всё-таки отказалась — no_answer с ближайшими темами;
- тест ждёт ответов человека — interrupt, а после ответов — проверка и запись слабых тем.

Каждый узел, который зовёт LLM, деградирует мягко: упал роутер — считаем запрос вопросом,
упал грейдер — пускаем к генерации, упал rewrite — честный отказ.
"""

import asyncio
import logging
import random
import re
import uuid
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from langchain_core.messages import AIMessage, RemoveMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.graph import END
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt

from app.answer.citations import is_refusal, render_citations
from app.answer.prompts import NO_ANSWER, answer_messages
from app.config import Settings
from app.graph import memory, prompts, rules
from app.graph.state import AgentContext, AgentState, GradeDecision, Quiz, RouteDecision
from app.llm.client import Completion, LLMClient, LLMError
from app.llm.images import ImageGenerator
from app.retrieval.search import Retriever
from app.retrieval.store import ScoredChunk

log = logging.getLogger(__name__)

OUT_OF_SCOPE = (
    "Я помогаю только с учебными материалами: отвечаю на вопросы по загруженным лекциям "
    "со ссылками на страницы и составляю тесты. Спросите что-нибудь по курсу."
)


class AgentNodes:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLMClient,
        settings: Settings,
        list_sections: Callable[[str | None], list[dict[str, Any]]] | None = None,
        images: ImageGenerator | None = None,
    ):
        self.retriever = retriever
        self.llm = llm
        self.s = settings
        self.list_sections = list_sections or (lambda course: [])
        self.images = images

    # ---------- память ----------

    def needs_summary(self, state: AgentState) -> str:
        messages = state.get("messages", [])
        if (
            len(messages) > self.s.summary_keep_messages
            and count_tokens_approximately(messages) > self.s.summary_trigger_tokens
        ):
            return "summarize"
        return "route"

    async def summarize(self, state: AgentState, runtime: Runtime[AgentContext]) -> dict:
        """Пакетная суммаризация: старая часть диалога сжимается разом, когда история перевалила
        за порог, а не на каждом ходу. Так префикс промпта стабилен между ходами и кэшируется."""
        _status(runtime, "summarize", "Сжимаю историю диалога")
        messages = state["messages"]
        old = messages[: -self.s.summary_keep_messages]
        try:
            c = await self.llm.complete(prompts.summary_messages(state.get("summary", ""), old), max_tokens=1024)
        except LLMError as e:
            log.warning("summarize failed, keep full history: %s", e)
            return {}
        return {"summary": c.text, "messages": [RemoveMessage(id=m.id) for m in old]}

    # ---------- маршрутизация ----------

    async def route(self, state: AgentState, runtime: Runtime[AgentContext]) -> dict:
        _status(runtime, "route", "Разбираю запрос")
        messages = state.get("messages", [])
        question = str(messages[-1].content).strip() if messages else ""
        history = messages[:-1]
        turn = {  # сброс полей прошлого хода
            "question": question,
            "query": question,
            "documents": [],
            "rewrite_count": 0,
            "answer": "",
            "citations": [],
            "no_answer": False,
            "quiz": None,
            "quiz_answers": None,
            "topic": None,
            "follow_up": False,
            "images": [],
        }

        decided = rules.classify(question)
        if decided:
            route, topic = decided
            by = "rules"
        else:
            try:
                decision, _ = await self.llm.complete_json(
                    prompts.route_messages(question, prompts.history_text(history, state.get("summary", ""))),
                    RouteDecision,
                    max_tokens=1024,
                )
                route, topic, by = decision.route, decision.topic, "llm"
            except (LLMError, ValueError) as e:
                log.warning("route llm failed, fallback to question: %s", e)
                route, topic, by = "question", None, "fallback"

        follow_up = route == "question" and rules.is_follow_up(question, bool(history))
        runtime.stream_writer({"type": "route", "route": route, "by": by, "follow_up": follow_up})
        return {**turn, "route": route, "topic": topic, "follow_up": follow_up}

    def after_route(self, state: AgentState) -> str:
        route = state["route"]
        if route == "question":
            # «а что насчёт второго?» искать бесполезно — сначала раскрыть отсылку
            return "rewrite" if state.get("follow_up") else "retrieve"
        return {
            "quiz": "make_quiz",
            "draw": "illustrate",
            "chitchat": "chitchat",
            "out_of_scope": "out_of_scope",
            "progress": "progress",
        }[route]

    # ---------- поиск и оценка ----------

    async def retrieve(self, state: AgentState, runtime: Runtime[AgentContext]) -> dict:
        _status(runtime, "retrieve", f"Ищу в материалах: «{state['query']}»")
        hits = await asyncio.to_thread(self.retriever.search, state["query"], self.s.top_k, runtime.context.course)
        return {"documents": [h.to_dict() for h in hits]}

    async def grade(self, state: AgentState, runtime: Runtime[AgentContext]) -> dict:
        docs = _docs(state)
        if not docs:
            runtime.stream_writer({"type": "grade", "grade": "irrelevant", "best": None, "by": "empty"})
            return {"grade": "irrelevant"}

        best = docs[0].relevance
        if self.retriever.reranker is None:
            # без реранкера шкала косинуса плохо разделяет «есть/нет», решает генерация
            grade = "relevant" if best >= self.s.min_score else "irrelevant"
            by = "cosine"
        elif best >= self.s.relevance_threshold:
            grade, by = "relevant", "reranker"
        elif best < self.s.relevance_floor:
            grade, by = "irrelevant", "reranker"
        else:
            # серая зона: реранкер не уверен — коротко спрашиваем LLM
            _status(runtime, "grade", "Проверяю, подходит ли найденное")
            try:
                decision, _ = await self.llm.complete_json(
                    prompts.grade_messages(state.get("query") or state["question"], docs[:3]), GradeDecision,
                    max_tokens=1024,
                )
                grade, by = ("relevant" if decision.relevant else "irrelevant"), "llm"
            except (LLMError, ValueError) as e:
                log.warning("grade llm failed, let generate decide: %s", e)
                grade, by = "relevant", "fallback"

        runtime.stream_writer({"type": "grade", "grade": grade, "best": round(best, 4), "by": by})
        if grade == "relevant" and self.retriever.reranker is not None:
            # в промпт — только то, что реранкер счёл хоть сколько-то относящимся к вопросу
            kept = [d for d in docs if d.relevance >= self.s.relevance_floor] or docs[:1]
            return {"grade": grade, "documents": [d.to_dict() for d in kept]}
        return {"grade": grade}

    def after_grade(self, state: AgentState) -> str:
        if state["grade"] == "relevant":
            return "generate"
        if state.get("rewrite_count", 0) < self.s.max_rewrites:
            return "rewrite"
        return "no_answer"

    async def rewrite(self, state: AgentState, runtime: Runtime[AgentContext]) -> Command:
        _status(runtime, "rewrite", "Переформулирую запрос")
        messages = state.get("messages", [])
        tried = [state["query"]] if state.get("documents") or state.get("rewrite_count") else []
        sections = list(dict.fromkeys(d["section"] for d in state.get("documents", []) if d.get("section")))
        try:
            c = await self.llm.complete(
                prompts.rewrite_messages(
                    state["question"],
                    prompts.history_text(messages[:-1], state.get("summary", "")),
                    tried,
                    sections,
                ),
                max_tokens=1024,
            )
        except LLMError as e:
            log.warning("rewrite failed: %s", e)
            return Command(goto="no_answer")
        query = _one_line(c.text) or state["question"]
        runtime.stream_writer({"type": "rewrite", "query": query})
        return Command(
            update={"query": query, "rewrite_count": state.get("rewrite_count", 0) + 1},
            goto="retrieve",
        )

    # ---------- ответ ----------

    async def generate(self, state: AgentState, runtime: Runtime[AgentContext]) -> Command:
        _status(runtime, "generate", "Пишу ответ")
        docs = _docs(state)
        history = prompts.history_text(state["messages"][:-1], state.get("summary", ""), limit=4)
        messages = answer_messages(state["question"], docs, history, state.get("query"))
        completion: Completion | None = None
        try:
            async for item in self.llm.stream(messages):
                if isinstance(item, Completion):
                    completion = item
                else:
                    runtime.stream_writer({"type": "token", "text": item})
        except LLMError as e:
            # Модель недоступна, но поиск сработал — показываем, где ответ, а не ошибку
            log.error("generate failed: %s", e)
            text = "Модель сейчас недоступна, но вот где это есть в материалах:\n" + "\n".join(
                f"- {_label(d)}" for d in docs[:3]
            )
            return Command(
                update={"answer": text, "citations": [_citation(d) for d in docs[:3]], "messages": [AIMessage(text)]},
                goto=END,
            )

        if completion is None or is_refusal(completion.text):
            return Command(update={"no_answer": True}, goto="no_answer")
        text, citations = render_citations(completion.text, docs)
        return Command(
            update={
                "answer": text,
                "citations": [{**asdict(c), "label": c.label} for c in citations],
                "messages": [AIMessage(text)],
            },
            goto=END,
        )

    async def no_answer(self, state: AgentState, runtime: Runtime[AgentContext]) -> dict:
        _status(runtime, "no_answer", "В материалах ответа нет")
        docs = sorted(_docs(state), key=lambda d: d.relevance, reverse=True)
        if self.retriever.reranker is not None:
            # «стих про кота» → «Понятие заряда» с реранком 0,0002 — не «ближайшая тема», а шум
            docs = [d for d in docs if d.relevance >= self.s.nearest_min_relevance]
        nearest, seen = [], set()
        for d in docs:
            key = (d.file_id, d.section or d.page)
            if key not in seen:
                seen.add(key)
                nearest.append(d)
            if len(nearest) == 3:
                break
        text = NO_ANSWER
        if nearest:
            text += "\n\nБлиже всего в материалах:\n" + "\n".join(f"- {_label(d)}" for d in nearest)
        elif not self.retriever.store.exists():
            text += " Материалы ещё не загружены."
        return {
            "answer": text,
            "no_answer": True,
            "citations": [_citation(d) for d in nearest],
            "messages": [AIMessage(text)],
        }

    # ---------- иллюстрация ----------

    async def illustrate(self, state: AgentState, runtime: Runtime[AgentContext]) -> dict:
        """Рисует учебную схему по теме курса — только если тема есть в лекциях.

        Картинка строится по тексту найденного фрагмента, и подпись прямо говорит, что это иллюстрация
        по описанию, а не рисунок из лекции: процитировать картинку как источник нельзя."""
        if self.images is None:
            text = "Рисовать я пока не могу: генерация картинок работает через GigaChat, а он не настроен."
            return {"answer": text, "messages": [AIMessage(text)]}

        topic = state.get("topic") or state["question"]
        _status(runtime, "illustrate", f"Ищу в лекциях тему «{topic}»")
        hits = await asyncio.to_thread(self.retriever.search, topic, 3, runtime.context.course)
        if self.retriever.reranker is not None:
            # рисуем только уверенно найденное: «нарисуй кота» не должно стать иллюстрацией к курсу физики
            hits = [h for h in hits if h.relevance >= self.s.relevance_threshold]
        if not hits:
            text = f"В лекциях нет темы «{topic}», а рисую я только иллюстрации к материалам курса."
            return {"answer": text, "no_answer": True, "messages": [AIMessage(text)]}

        source = hits[0]
        _status(runtime, "illustrate", "Рисую схему — это около 15 секунд")
        prompt = (
            f"Нарисуй учебную иллюстрацию: {topic}.\n"
            f"Что должно быть изображено — по тексту лекции («{source.section or topic}»):\n{source.text[:700]}"
        )
        try:
            image = await self.images.generate(prompt)
        except LLMError as e:
            log.warning("illustrate failed: %s", e)
            text = "Не получилось нарисовать — генератор картинок сейчас недоступен. Попробуйте позже."
            return {"answer": text, "messages": [AIMessage(text)]}

        image_id = uuid.uuid4().hex
        await asyncio.to_thread(_save_image, self.s.images_dir, image_id, image.data)
        text = (
            f"Иллюстрация к теме «{topic}». Нарисована по описанию из лекции — это не рисунок из самой лекции.\n"
            f"Описание: {_label(source)}"
        )
        return {
            "answer": text,
            "images": [{"image_id": image_id, "topic": topic}],
            "citations": [_citation(source)],
            "messages": [AIMessage(text)],
        }

    # ---------- болтовня, не по теме, прогресс ----------

    async def chitchat(self, state: AgentState, runtime: Runtime[AgentContext]) -> dict:
        question = state["question"]
        text = rules.small_talk_reply(question)
        if text is None:
            try:
                history = prompts.history_text(state["messages"][:-1], state.get("summary", ""), limit=4)
                text = (await self.llm.complete(prompts.chitchat_messages(question, history), max_tokens=512)).text
            except LLMError:
                text = rules.small_talk_reply("привет")
        return {"answer": text, "messages": [AIMessage(text)]}

    async def out_of_scope(self, state: AgentState, runtime: Runtime[AgentContext]) -> dict:
        return {"answer": OUT_OF_SCOPE, "messages": [AIMessage(OUT_OF_SCOPE)]}

    async def progress(self, state: AgentState, runtime: Runtime[AgentContext]) -> dict:
        topics = await memory.weak_topics(runtime.store, runtime.context.user_id)
        if not topics:
            text = "Ошибок в тестах пока нет. Попросите тест по любой теме — например, «составь тест по закону Кулона»."
        else:
            lines = []
            for i, t in enumerate(topics, start=1):
                where = "; ".join(f"{s['filename']}, стр. {s['page']}" for s in t.get("sources", []))
                lines.append(f"{i}. {t['topic']} — ошибок: {t['mistakes']}" + (f" ({where})" if where else ""))
            text = (
                "Темы, в которых были ошибки:\n" + "\n".join(lines)
                + f"\n\nЧтобы повторить, попросите: «составь тест по теме {topics[0]['topic']}»."
            )
        return {"answer": text, "messages": [AIMessage(text)]}

    # ---------- тест ----------

    async def make_quiz(self, state: AgentState, runtime: Runtime[AgentContext]) -> Command:
        ctx = runtime.context
        weak = await memory.weak_topics(runtime.store, ctx.user_id)
        topic = state.get("topic") or (weak[0]["topic"] if weak else await asyncio.to_thread(self._any_section, ctx.course))
        if not topic:
            return Command(update={"documents": []}, goto="no_answer")

        _status(runtime, "make_quiz", f"Составляю тест по теме «{topic}»")
        hits = await asyncio.to_thread(self.retriever.search, topic, max(self.s.top_k, 8), ctx.course)
        if self.retriever.reranker is not None:
            hits = [h for h in hits if h.relevance >= self.s.relevance_floor]
        if not hits:
            return Command(update={"documents": [], "topic": topic}, goto="no_answer")

        try:
            quiz, _ = await self.llm.complete_json(
                prompts.quiz_messages(topic, hits, self.s.quiz_questions, [w["topic"] for w in weak]),
                Quiz,
                max_tokens=6000,
            )
        except (LLMError, ValueError) as e:
            log.warning("quiz generation failed: %s", e)
            text = "Не получилось составить тест — модель вернула некорректный ответ. Попробуйте ещё раз."
            return Command(update={"answer": text, "messages": [AIMessage(text)]}, goto=END)

        questions = _valid_questions(quiz, hits)[: self.s.quiz_questions]
        if not questions:
            text = "Не получилось составить тест по этой теме из найденных фрагментов."
            return Command(update={"answer": text, "messages": [AIMessage(text)]}, goto=END)
        intro = f"Тест по теме «{topic}»: {len(questions)} вопросов. Отметьте ответы."
        return Command(
            update={
                "topic": topic,
                "documents": [h.to_dict() for h in hits],
                "quiz": {"topic": topic, "questions": questions},
                "messages": [AIMessage(intro)],
            },
            goto="ask_quiz",
        )

    async def ask_quiz(self, state: AgentState, runtime: Runtime[AgentContext]) -> dict:
        """Пауза до ответов человека. Узел перезапускается с начала после resume, поэтому
        генерация теста вынесена в make_quiz — иначе тест пересоздавался бы при каждом ответе."""
        quiz = state["quiz"]
        answers = interrupt({
            "type": "quiz",
            "topic": quiz["topic"],
            "questions": [{"question": q["question"], "options": q["options"]} for q in quiz["questions"]],
        })
        return {"quiz_answers": _normalize_answers(answers, len(quiz["questions"]))}

    async def check_quiz(self, state: AgentState, runtime: Runtime[AgentContext]) -> dict:
        quiz = state["quiz"]
        answers = state.get("quiz_answers") or []
        lines, results, citations = [], [], []
        for i, (q, a) in enumerate(zip(quiz["questions"], answers, strict=True), start=1):
            ok = a == q["correct"]
            src = q["source_ref"]
            label = f"[{src['filename']}, стр. {src['page']}]"
            results.append({"topic": q["topic"], "correct": ok, "source": {"filename": src["filename"], "page": src["page"]}})
            if ok:
                lines.append(f"{i}. ✅ {q['question']}")
            else:
                given = q["options"][a] if a is not None and a < len(q["options"]) else "нет ответа"
                lines.append(
                    f"{i}. ❌ {q['question']}\n   Ваш ответ: {given}. Правильно: {q['options'][q['correct']]}.\n"
                    f"   {q['explanation']} {label}"
                )
                citations.append({**src, "label": label})
        score = sum(r["correct"] for r in results)
        weakened = await memory.record_answers(runtime.store, runtime.context.user_id, results) if runtime.store else []
        text = f"Результат: {score} из {len(results)}.\n\n" + "\n".join(lines)
        if weakened:
            text += "\n\nДобавил в темы для повторения: " + ", ".join(dict.fromkeys(weakened)) + "."
        return {
            "answer": text,
            "citations": citations,
            "quiz": None,
            "messages": [AIMessage(text)],
        }

    def _any_section(self, course: str | None) -> str | None:
        sections = self.list_sections(course)
        return random.choice(sections)["section"] if sections else None


# ---------- помощники ----------


def _status(runtime: Runtime, node: str, text: str) -> None:
    runtime.stream_writer({"type": "status", "node": node, "text": text})


def _save_image(directory, image_id: str, data: bytes) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{image_id}.jpg").write_bytes(data)


def _docs(state: AgentState) -> list[ScoredChunk]:
    return [ScoredChunk.from_dict(d) for d in state.get("documents", [])]


def _label(d: ScoredChunk) -> str:
    where = f"{d.filename}, стр. {d.page}"
    return f"«{d.section}» — {where}" if d.section else where


def _citation(d: ScoredChunk) -> dict[str, Any]:
    return {
        "filename": d.filename,
        "page": d.page,
        "file_id": d.file_id,
        "snippet": d.text[:300],
        "label": f"[{d.filename}, стр. {d.page}]",
    }


def _one_line(text: str) -> str:
    line = next((ln for ln in text.strip().splitlines() if ln.strip()), "")
    return re.sub(r"^(запрос:\s*)", "", line.strip().strip("«»\"'`"), flags=re.IGNORECASE).strip()


def _valid_questions(quiz: Quiz, docs: list[ScoredChunk]) -> list[dict[str, Any]]:
    """Отбрасывает вопросы с битым индексом ответа или ссылкой на несуществующий документ
    и перемешивает варианты: модели любят ставить правильный ответ первым."""
    result = []
    for q in quiz.questions:
        options = [o.strip() for o in q.options if o.strip()]
        if len(options) < 3 or not 0 <= q.correct < len(options) or not 1 <= q.source <= len(docs):
            continue
        correct_text = options[q.correct]
        rng = random.Random(q.question)  # детерминированно: один и тот же вопрос — один и тот же порядок
        rng.shuffle(options)
        d = docs[q.source - 1]
        result.append({
            "question": q.question.strip(),
            "options": options,
            "correct": options.index(correct_text),
            "explanation": q.explanation.strip(),
            "topic": q.topic.strip() or quiz.topic,
            "source_ref": {"filename": d.filename, "page": d.page, "file_id": d.file_id},
        })
    return result


def _normalize_answers(answers: Any, n: int) -> list[int | None]:
    if isinstance(answers, dict):
        answers = answers.get("answers", [])
    if not isinstance(answers, list):
        answers = []
    result: list[int | None] = []
    for a in list(answers)[:n]:
        result.append(a if isinstance(a, int) and a >= 0 else None)
    return result + [None] * (n - len(result))
