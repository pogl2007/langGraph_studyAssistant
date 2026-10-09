"""Маршруты графа с замоканной LLM и настоящим поиском по маленькому набору документов."""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from app.answer.prompts import NO_ANSWER
from app.graph import memory
from app.graph.builder import build_graph
from app.graph.nodes import OUT_OF_SCOPE, AgentNodes
from app.graph.state import AgentContext

from .conftest import LLMDown, ScriptedLLM, make_pdf, make_settings

LECTURE = [
    "# Закон Кулона\nСила взаимодействия двух неподвижных точечных зарядов пропорциональна произведению "
    "зарядов и обратно пропорциональна квадрату расстояния между ними.",
    "# Теорема Гаусса\nПоток вектора напряженности электрического поля через замкнутую поверхность равен "
    "суммарному заряду внутри поверхности, деленному на электрическую постоянную.",
    "# Сила Лоренца\nНа заряд, движущийся в магнитном поле, действует сила Лоренца, перпендикулярная скорости заряда.",
]

GAUSS_Q = "Что говорит теорема Гаусса о потоке напряженности?"


@pytest.fixture
def make_graph(indexer, retriever):
    indexer.index_pdf("lecture.pdf", make_pdf(LECTURE), course="physics")
    stores = {}

    def build(llm, **overrides):
        nodes = AgentNodes(retriever, llm, make_settings(**overrides), list_sections=retriever.store.list_sections)
        stores["memory"] = InMemoryStore()
        return build_graph(nodes, checkpointer=InMemorySaver(), store=stores["memory"])

    build.stores = stores
    return build


async def run(graph, payload, thread="t1", user="u1"):
    config = {"configurable": {"thread_id": thread}, "recursion_limit": 30}
    if isinstance(payload, str):
        payload = {"messages": [HumanMessage(payload)]}
    path, events = [], []
    async for mode, chunk in graph.astream(
        payload, config, context=AgentContext(user_id=user, course="physics"), stream_mode=["updates", "custom"]
    ):
        if mode == "updates":
            path.extend(chunk.keys())
        else:
            events.append(chunk)
    state = (await graph.aget_state(config)).values
    return path, state, events


async def test_question_answered_with_citation(make_graph):
    llm = ScriptedLLM(answer="Поток равен заряду внутри поверхности, деленному на постоянную [1].")
    path, state, events = await run(make_graph(llm), GAUSS_Q)

    assert path == ["route", "retrieve", "grade", "generate"]
    assert state["answer"] == "Поток равен заряду внутри поверхности, деленному на постоянную [lecture.pdf, стр. 2]."
    assert state["citations"][0]["page"] == 2
    assert llm.kinds() == ["answer"]  # маршрут по правилам, грейд по реранкеру — LLM только для ответа
    tokens = "".join(e["text"] for e in events if e["type"] == "token")
    assert tokens.strip().endswith("[1].")  # токены стримятся до подстановки цитат
    assert isinstance(state["messages"][-1], AIMessage)


async def test_irrelevant_question_rewrites_then_refuses(make_graph):
    llm = ScriptedLLM(rewrite=["рецепт борща со свеклой", "как варить борщ"])
    path, state, _ = await run(make_graph(llm, max_rewrites=2), "Какой рецепт у борща?")

    assert path == ["route", "retrieve", "grade", "rewrite", "retrieve", "grade", "rewrite", "retrieve", "grade",
                    "no_answer"]
    assert state["rewrite_count"] == 2
    assert llm.kinds() == ["rewrite", "rewrite"]
    assert state["answer"].startswith(NO_ANSWER)
    assert state["no_answer"] is True


async def test_rewrite_limit_is_respected(make_graph):
    llm = ScriptedLLM(rewrite=lambda m: "борщ")
    path, state, _ = await run(make_graph(llm, max_rewrites=1), "Какой рецепт у борща?")

    assert path.count("rewrite") == 1
    assert path[-1] == "no_answer"


async def test_rewrite_finds_answer(make_graph):
    llm = ScriptedLLM(rewrite="теорема Гаусса поток напряженности", answer="Поток вектора равен заряду [1].")
    path, state, events = await run(make_graph(llm), "Объясни ту формулу")

    assert path == ["route", "retrieve", "grade", "rewrite", "retrieve", "grade", "generate"]
    assert state["query"] == "теорема Гаусса поток напряженности"
    assert "стр. 2" in state["answer"]
    assert {"type": "rewrite", "query": "теорема Гаусса поток напряженности"} in events


async def test_follow_up_is_rewritten_with_history_before_search(make_graph):
    llm = ScriptedLLM(
        answer=["Поток равен заряду [1].", "Сила обратно пропорциональна квадрату расстояния [1]."],
        rewrite="закон Кулона сила взаимодействия зарядов",
    )
    graph = make_graph(llm)
    await run(graph, GAUSS_Q)
    path, state, _ = await run(graph, "а закон Кулона?")

    assert path == ["route", "rewrite", "retrieve", "grade", "generate"]
    rewrite_prompt = next(m for k, m in llm.calls if k == "rewrite")[1]["content"]
    assert "теорема Гаусса" in rewrite_prompt  # в переписывание ушла история
    assert "стр. 1" in state["answer"]


async def test_gray_zone_asks_llm_grader(make_graph):
    llm = ScriptedLLM(grade=json.dumps({"relevant": True, "reason": "есть"}), answer="Кулон открыл закон [1].")
    path, state, _ = await run(make_graph(llm, relevance_threshold=0.9, relevance_floor=0.1), "Кто такой Кулон?")

    assert path == ["route", "retrieve", "grade", "generate"]
    assert llm.kinds() == ["grade", "answer"]


async def test_model_refusal_goes_to_no_answer_with_nearest_topics(make_graph):
    llm = ScriptedLLM(answer=NO_ANSWER)
    path, state, _ = await run(make_graph(llm), GAUSS_Q)

    assert path == ["route", "retrieve", "grade", "generate", "no_answer"]
    assert "Теорема Гаусса" in state["answer"]  # ближайшая тема
    assert state["citations"]


async def test_llm_down_shows_sources_instead_of_error(make_graph):
    llm = ScriptedLLM(answer=LLMDown("all providers failed"))
    path, state, _ = await run(make_graph(llm), GAUSS_Q)

    assert path[-1] == "generate"
    assert "Модель сейчас недоступна" in state["answer"]
    assert state["citations"][0]["page"] == 2


async def test_router_failure_falls_back_to_question(make_graph):
    llm = ScriptedLLM(route=LLMDown("timeout"), answer="Поток равен заряду [1].")
    path, state, events = await run(make_graph(llm), "Поток напряженности теорема Гаусса")

    assert path[:2] == ["route", "retrieve"]
    assert any(e.get("by") == "fallback" for e in events if e["type"] == "route")


async def test_small_talk_without_llm(make_graph):
    llm = ScriptedLLM()
    path, state, _ = await run(make_graph(llm), "Привет!")

    assert path == ["route", "chitchat"]
    assert llm.calls == []
    assert "Привет" in state["answer"]


async def test_out_of_scope_by_llm_router(make_graph):
    llm = ScriptedLLM(route=json.dumps({"route": "out_of_scope", "topic": None}))
    path, state, _ = await run(make_graph(llm), "Напиши стих про кота")

    assert path == ["route", "out_of_scope"]
    assert state["answer"] == OUT_OF_SCOPE


def quiz_json(n: int = 3) -> str:
    return json.dumps({
        "topic": "Закон Кулона",
        "questions": [
            {
                "question": f"Вопрос {i} про закон Кулона",
                "options": ["обратно квадрату расстояния", "прямо расстоянию", "не зависит", "экспоненциально"],
                "correct": 0,
                "explanation": "Так сказано в лекции.",
                "source": 1,
                "topic": "закон Кулона" if i == 1 else f"подтема {i}",
            }
            for i in range(1, n + 1)
        ],
    })


async def test_quiz_interrupts_checks_answers_and_remembers_weak_topics(make_graph):
    llm = ScriptedLLM(quiz=quiz_json())
    graph = make_graph(llm)

    path, state, _ = await run(graph, "Составь тест по теме Закон Кулона")
    assert path == ["route", "make_quiz", "__interrupt__"]
    pending = (await graph.aget_state({"configurable": {"thread_id": "t1"}})).tasks[0].interrupts[0].value
    assert pending["type"] == "quiz"
    assert "correct" not in json.dumps(pending)  # правильные ответы клиенту не уходят
    assert len(pending["questions"]) == 3

    correct = [q["correct"] for q in state["quiz"]["questions"]]
    answers = [correct[0], (correct[1] + 1) % 4, None]  # верно, неверно, без ответа
    path, state, _ = await run(graph, Command(resume=answers))

    assert path == ["ask_quiz", "check_quiz"]
    assert state["answer"].startswith("Результат: 1 из 3.")
    assert "[lecture.pdf, стр. 1]" in state["answer"]
    weak = await memory.weak_topics(make_graph.stores["memory"], "u1")
    assert [w["topic"] for w in weak] == ["подтема 3", "подтема 2"] or {w["topic"] for w in weak} == {
        "подтема 2", "подтема 3"}

    # слабые темы видны в следующем диалоге того же пользователя
    path, state, _ = await run(graph, "Что мне повторить?", thread="t2")
    assert path == ["route", "progress"]
    assert "подтема 2" in state["answer"]


async def test_quiz_options_are_shuffled_but_answer_tracked(make_graph):
    llm = ScriptedLLM(quiz=quiz_json(3))
    graph = make_graph(llm)
    _, state, _ = await run(graph, "Составь тест по теме Закон Кулона")

    for q in state["quiz"]["questions"]:
        assert q["options"][q["correct"]] == "обратно квадрату расстояния"
    assert any(q["correct"] != 0 for q in state["quiz"]["questions"])


async def test_quiz_drops_questions_with_broken_source(make_graph):
    data = json.loads(quiz_json(3))
    data["questions"][1]["source"] = 99
    data["questions"][2]["correct"] = 7
    graph = make_graph(ScriptedLLM(quiz=json.dumps(data)))
    _, state, _ = await run(graph, "Составь тест по теме Закон Кулона")

    assert len(state["quiz"]["questions"]) == 1


async def test_correct_answer_heals_weak_topic(make_graph):
    graph = make_graph(ScriptedLLM(quiz=[quiz_json(1), quiz_json(1)]))
    store = make_graph.stores["memory"]

    _, state, _ = await run(graph, "Составь тест по теме Закон Кулона")
    wrong = (state["quiz"]["questions"][0]["correct"] + 1) % 4
    await run(graph, Command(resume=[wrong]))
    assert [w["topic"] for w in await memory.weak_topics(store, "u1")] == ["закон Кулона"]

    _, state, _ = await run(graph, "Составь тест по теме Закон Кулона", thread="t2")
    await run(graph, Command(resume=[state["quiz"]["questions"][0]["correct"]]), thread="t2")
    assert await memory.weak_topics(store, "u1") == []


async def test_quiz_without_topic_uses_weak_topic(make_graph):
    graph = make_graph(ScriptedLLM(quiz=quiz_json(1)))
    store = make_graph.stores["memory"]
    await memory.record_answers(store, "u1", [{"topic": "Теорема Гаусса", "correct": False}])

    _, state, _ = await run(graph, "Проверь меня")
    assert state["topic"] == "Теорема Гаусса"


async def test_long_history_is_summarized_in_batch(make_graph):
    llm = ScriptedLLM(summary="Студент спрашивал про закон Кулона и теорему Гаусса.", answer="Поток равен заряду [1].")
    graph = make_graph(llm, summary_trigger_tokens=60, summary_keep_messages=2)
    long = "очень длинное сообщение про электростатику " * 10
    history = [HumanMessage(long), AIMessage(long), HumanMessage(long), AIMessage("Кратко."), HumanMessage(GAUSS_Q)]
    path, state, _ = await run(graph, {"messages": history})

    assert path[0] == "summarize"
    assert state["summary"].startswith("Студент спрашивал")
    # остались 2 последних сообщения и новый ответ
    assert len(state["messages"]) == 3
    assert llm.kinds() == ["summary", "answer"]

    # следующий короткий ход не суммаризирует снова: порог считается по истории, а не по ходам
    path, _, _ = await run(graph, "Привет!")
    assert "summarize" not in path


async def test_graph_renders_to_mermaid(make_graph):
    mermaid = make_graph(ScriptedLLM()).get_graph().draw_mermaid()
    for node in ("route", "retrieve", "grade", "rewrite", "generate", "no_answer", "make_quiz", "ask_quiz"):
        assert node in mermaid
