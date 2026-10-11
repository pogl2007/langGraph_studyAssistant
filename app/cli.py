"""Командная строка: индексация папки, отладка поиска и вопрос агенту без API и интерфейса.

    python -m app.cli index "C:/lectures/physics" --course physics
    python -m app.cli search "теорема Гаусса" --course physics --mode dense --no-rerank
    python -m app.cli ask "Что такое напряжённость поля?" --course physics
    python -m app.cli graph            # схема графа в Mermaid
"""

import argparse
import asyncio
import logging
import sys
import time
import uuid
from pathlib import Path


def cmd_index(args: argparse.Namespace) -> None:
    from app.services import get_indexer

    files = sorted(p for p in Path(args.folder).glob(args.pattern) if p.suffix.lower() == ".pdf")
    if not files:
        sys.exit(f"нет PDF в {args.folder}")
    indexer = get_indexer()
    total = time.perf_counter()
    for f in files:
        r = indexer.index_pdf(f.name, f.read_bytes(), course=args.course, force=args.force)
        print(f"{r.status:9} {r.filename[:60]:60} стр={r.pages:3} чанков={r.chunks:3} {r.seconds:6.1f} с")
    print(f"готово: {len(files)} файлов за {time.perf_counter() - total:.0f} с")


def cmd_search(args: argparse.Namespace) -> None:
    from app.services import get_retriever

    retriever = get_retriever(mode=args.mode, rerank=not args.no_rerank)
    retriever.search("прогрев", 1, args.course)  # первый вызов грузит модели — не мерить его
    started = time.perf_counter()
    hits = retriever.search(args.query, args.top_k, args.course)
    print(f"{len(hits)} результатов за {time.perf_counter() - started:.2f} с ({retriever.mode}, реранкер={bool(retriever.reranker)})")
    for h in hits:
        rr = f" rerank={h.rerank_score:.3f}" if h.rerank_score is not None else ""
        print(f"  score={h.score:.3f}{rr} | {h.filename[:40]} стр.{h.page} | {h.section} | {h.text[:90]!r}")


async def _ask(args: argparse.Namespace) -> None:
    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.store.memory import InMemoryStore

    from app.config import get_settings
    from app.graph.state import AgentContext
    from app.llm.client import LLMClient
    from app.services import build_agent, build_images

    s = get_settings()
    llm = LLMClient.from_settings(s)
    images = build_images(s)
    graph = build_agent(llm, checkpointer=InMemorySaver(), store=InMemoryStore(), images=images)
    config = {"configurable": {"thread_id": uuid.uuid4().hex}, "recursion_limit": s.recursion_limit}
    try:
        async for mode, chunk in graph.astream(
            {"messages": [HumanMessage(args.question)]},
            config,
            context=AgentContext(course=args.course),
            stream_mode=["updates", "custom"],
        ):
            if mode == "custom" and chunk["type"] == "token":
                print(chunk["text"], end="", flush=True)
            elif mode == "custom" and chunk["type"] != "status":
                print(f"\n  · {chunk}", flush=True)
            elif mode == "updates":
                print(f"\n[{', '.join(chunk)}]", flush=True)
        values = (await graph.aget_state(config)).values
        print("\n\n=== Ответ ===\n" + values.get("answer", ""))
        for img in values.get("images", []):
            print(f"Картинка: {s.images_dir / (img['image_id'] + '.jpg')}")
    finally:
        await llm.aclose()
        if images is not None:
            await images.aclose()


def cmd_graph(args: argparse.Namespace) -> None:
    from app.config import get_settings
    from app.graph.builder import build_graph
    from app.graph.nodes import AgentNodes

    class _Stub:  # для схемы не нужны ни модели, ни Qdrant
        reranker = store = None

    print(build_graph(AgentNodes(_Stub(), None, get_settings())).get_graph().draw_mermaid())


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
    p = argparse.ArgumentParser(prog="python -m app.cli")
    sub = p.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("index", help="проиндексировать PDF из папки")
    i.add_argument("folder")
    i.add_argument("--course", default="default")
    i.add_argument("--pattern", default="*.pdf")
    i.add_argument("--force", action="store_true", help="переиндексировать даже без изменений")
    i.set_defaults(func=cmd_index)

    s = sub.add_parser("search", help="отладка поиска")
    s.add_argument("query")
    s.add_argument("--course")
    s.add_argument("--top-k", type=int, default=5)
    s.add_argument("--mode", choices=["hybrid", "dense"])
    s.add_argument("--no-rerank", action="store_true")
    s.set_defaults(func=cmd_search)

    a = sub.add_parser("ask", help="вопрос агенту")
    a.add_argument("question")
    a.add_argument("--course")
    a.set_defaults(func=lambda args: asyncio.run(_ask(args)))

    g = sub.add_parser("graph", help="схема графа в Mermaid")
    g.set_defaults(func=cmd_graph)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
