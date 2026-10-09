"""Оценка поиска на golden set: dense → hybrid → + реранкер. LLM не нужна.

    python eval/retrieval_eval.py --course physics

Метрики по вопросам, у которых есть ответ в материалах:
- recall@k — доля вопросов, где хотя бы одна правильная страница попала в top-k
  (у вопроса может быть несколько правильных страниц: одни и те же слайды повторяются в разных лекциях);
- MRR@10 — среднее 1/ранг первой правильной страницы (0, если её нет в top-10);
- задержка поиска p50 / p95 (эмбеддинг запроса + Qdrant + реранкер).

Для вопросов без ответа — отделимость: где лежит лучший скор у вопросов с ответом и без.
"""

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.services import get_retriever  # noqa: E402

CONFIGS = [
    ("dense", "dense", False),
    ("hybrid (dense + BM25, RRF)", "hybrid", False),
    ("dense + реранкер", "dense", True),
    ("hybrid + реранкер", "hybrid", True),
]


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def first_hit(hits, expected: list[dict]) -> int | None:
    for rank, h in enumerate(hits, start=1):
        if any(h.filename.startswith(e["file"]) and h.page == e["page"] for e in expected):
            return rank
    return None


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(round(p * (len(values) - 1))))]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--course", default="physics")
    p.add_argument("--golden", default="eval/golden_set.jsonl")
    p.add_argument("--out", default="eval/results/retrieval")
    args = p.parse_args()
    logging.disable(logging.INFO)

    s = get_settings()
    questions = load(Path(args.golden))
    answerable = [q for q in questions if q["expected"]]
    unanswerable = [q for q in questions if not q["expected"]]

    rows, details = [], {}
    for name, mode, rerank in CONFIGS:
        retriever = get_retriever(mode=mode, rerank=rerank)
        retriever.search("прогрев моделей", 1, args.course)
        ranks, times, best_ok, best_no = [], [], [], []
        per_q = []
        for q in questions:
            started = time.perf_counter()
            hits = retriever.search(q["question"], 10, args.course)
            times.append(time.perf_counter() - started)
            best = hits[0].relevance if hits else 0.0
            if q["expected"]:
                rank = first_hit(hits, q["expected"])
                ranks.append(rank)
                best_ok.append(best)
            else:
                rank = None
                best_no.append(best)
            per_q.append({"id": q["id"], "rank": rank, "best": round(best, 4),
                          "top": [f"{h.filename[:12]}…стр.{h.page}" for h in hits[:3]]})
        n = len(ranks)
        row = {
            "config": name,
            "hit@1": sum(1 for r in ranks if r == 1) / n,
            "recall@5": sum(1 for r in ranks if r and r <= 5) / n,
            "recall@10": sum(1 for r in ranks if r) / n,
            "mrr@10": sum(1 / r for r in ranks if r) / n,
            "p50": percentile(times, 0.5),
            "p95": percentile(times, 0.95),
            "best_answerable_min": min(best_ok),
            "best_answerable_median": statistics.median(best_ok),
            "best_unanswerable_max": max(best_no) if best_no else None,
        }
        if rerank:
            row["accepted_answerable"] = sum(b >= s.relevance_threshold for b in best_ok) / n
            row["gray_answerable"] = sum(s.relevance_floor <= b < s.relevance_threshold for b in best_ok) / n
            row["rejected_unanswerable"] = sum(b < s.relevance_floor for b in best_no) / max(len(best_no), 1)
        rows.append(row)
        details[name] = per_q
        print(f"{name:28} recall@5={row['recall@5']:.3f} mrr@10={row['mrr@10']:.3f} p50={row['p50']:.2f}s", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "course": args.course,
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "chunking": s.chunking,
        "top_k_candidates_for_rerank": s.rerank_candidates,
        "embedding": s.embedding_model,
        "reranker": s.reranker_model,
        "threshold": s.relevance_threshold,
        "floor": s.relevance_floor,
    }
    out.with_suffix(".json").write_text(
        json.dumps({"meta": meta, "rows": rows, "details": details}, ensure_ascii=False, indent=2), "utf-8"
    )

    lines = [
        f"Golden set: {len(answerable)} вопросов с ответом и {len(unanswerable)} без ответа, курс «{args.course}», "
        f"чанкинг {s.chunking}, реранкер на {s.rerank_candidates} кандидатах, CPU.",
        "",
        "| Конфигурация | hit@1 | recall@5 | recall@10 | MRR@10 | p50 | p95 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['config']} | {r['hit@1']:.2f} | {r['recall@5']:.2f} | {r['recall@10']:.2f} | {r['mrr@10']:.3f} "
            f"| {r['p50']:.2f} с | {r['p95']:.2f} с |"
        )
    lines += [
        "",
        "Отделимость «есть ответ / нет ответа» по лучшему скору:",
        "",
        "| Конфигурация | мин. у вопросов с ответом | медиана у вопросов с ответом | макс. у вопросов без ответа |",
        "|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['config']} | {r['best_answerable_min']:.3f} | {r['best_answerable_median']:.3f} "
            f"| {r['best_unanswerable_max']:.3f} |"
        )
    rr = [r for r in rows if "accepted_answerable" in r]
    if rr:
        lines += ["", f"Пороги реранкера: ответ есть при скоре ≥ {s.relevance_threshold}, точно нет при < {s.relevance_floor}, "
                      "между ними — серая зона, решает LLM.", ""]
        lines += ["| Конфигурация | вопросы с ответом: принято сразу | в серой зоне | вопросы без ответа: отвергнуто сразу |",
                  "|---|---|---|---|"]
        for r in rr:
            lines.append(f"| {r['config']} | {r['accepted_answerable']:.0%} | {r['gray_answerable']:.0%} "
                         f"| {r['rejected_unanswerable']:.0%} |")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", "utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
