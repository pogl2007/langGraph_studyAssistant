"""Сборка графа агента.

    START ─┬─► summarize ─┐
           └──────────────┴─► route ─┬─► retrieve ─► grade ─┬─► generate ─┬─► END
                                     │      ▲               │             └─► no_answer ─► END
                                     ├─► rewrite ◄──────────┤ (не больше max_rewrites раз)
                                     │      └─► (ошибка) ───┴─► no_answer
                                     ├─► make_quiz ─► ask_quiz (interrupt) ─► check_quiz ─► END
                                     ├─► illustrate (схема по теме из лекций, GigaChat) ─► END
                                     ├─► chitchat / out_of_scope / progress ─► END

Точная схема генерируется из кода: `build_graph(...).get_graph().draw_mermaid()`.
"""

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from app.graph.nodes import AgentNodes
from app.graph.state import AgentContext, AgentState


def _transient(error: Exception) -> bool:
    # Qdrant или сеть до него моргнули — один повтор поиска дешевле, чем отказ пользователю
    name = type(error).__name__
    return name in {"ResponseHandlingException", "ConnectError", "ReadTimeout", "ConnectTimeout"}


def build_graph(nodes: AgentNodes, checkpointer=None, store=None, node_timeout: float | None = None):
    g = StateGraph(AgentState, context_schema=AgentContext)
    llm = {"timeout": node_timeout} if node_timeout else {}

    g.add_node("summarize", nodes.summarize, **llm)
    g.add_node("route", nodes.route, **llm)
    g.add_node("retrieve", nodes.retrieve, retry_policy=RetryPolicy(max_attempts=2, retry_on=_transient))
    g.add_node("grade", nodes.grade, **llm)
    g.add_node("rewrite", nodes.rewrite, destinations=("retrieve", "no_answer"), **llm)
    g.add_node("generate", nodes.generate, destinations=("no_answer", END), **llm)
    g.add_node("no_answer", nodes.no_answer)
    g.add_node("make_quiz", nodes.make_quiz, destinations=("ask_quiz", "no_answer", END), **llm)
    g.add_node("ask_quiz", nodes.ask_quiz)
    g.add_node("check_quiz", nodes.check_quiz)
    g.add_node("illustrate", nodes.illustrate, **llm)
    g.add_node("chitchat", nodes.chitchat, **llm)
    g.add_node("out_of_scope", nodes.out_of_scope)
    g.add_node("progress", nodes.progress)

    g.add_conditional_edges(START, nodes.needs_summary, ["summarize", "route"])
    g.add_edge("summarize", "route")
    g.add_conditional_edges(
        "route",
        nodes.after_route,
        ["retrieve", "rewrite", "make_quiz", "illustrate", "chitchat", "out_of_scope", "progress"],
    )
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", nodes.after_grade, ["generate", "rewrite", "no_answer"])
    g.add_edge("ask_quiz", "check_quiz")
    for node in ("no_answer", "check_quiz", "illustrate", "chitchat", "out_of_scope", "progress"):
        g.add_edge(node, END)

    return g.compile(checkpointer=checkpointer, store=store)
