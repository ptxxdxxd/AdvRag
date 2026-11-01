"""Retrieval metrics over a labeled question set.

Every question names one or more gold chunks: any of them fully answers it.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

from .index import Index

DEFAULT_QUESTIONS = Path(__file__).resolve().parent.parent / "evals" / "questions.jsonl"


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    gold: tuple[str, ...]
    evidence: str


def load_questions(path: Path | str = DEFAULT_QUESTIONS) -> list[Question]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            out.append(Question(d["id"], d["question"], tuple(d["gold"]), d["evidence"]))
    return out


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    """1.0 if any gold chunk is in the top k (a hit rate per question)."""
    return 1.0 if gold & set(ranked[:k]) else 0.0


def reciprocal_rank(ranked: list[str], gold: set[str], k: int = 10) -> float:
    for rank, cid in enumerate(ranked[:k], start=1):
        if cid in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: list[str], gold: set[str], k: int = 10) -> float:
    dcg = sum(1.0 / math.log2(r + 1) for r, cid in enumerate(ranked[:k], start=1) if cid in gold)
    ideal = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


@dataclass
class MethodResult:
    method: str
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr_at_10: float
    ndcg_at_10: float
    ms_per_query: float
    misses_at_5: list[str]


def evaluate(index: Index, questions: list[Question], methods=("bm25", "lsa", "hybrid")) -> list[MethodResult]:
    results = []
    for method in methods:
        r1 = r5 = r10 = mrr = ndcg = 0.0
        misses = []
        start = time.perf_counter()
        for q in questions:
            ranked = [c.id for c, _ in index.search(q.question, method, k=10)]
            gold = set(q.gold)
            r1 += recall_at_k(ranked, gold, 1)
            hit5 = recall_at_k(ranked, gold, 5)
            r5 += hit5
            r10 += recall_at_k(ranked, gold, 10)
            mrr += reciprocal_rank(ranked, gold)
            ndcg += ndcg_at_k(ranked, gold)
            if not hit5:
                misses.append(q.id)
        elapsed = time.perf_counter() - start
        n = len(questions)
        results.append(
            MethodResult(method, r1 / n, r5 / n, r10 / n, mrr / n, ndcg / n, 1000 * elapsed / n, misses)
        )
    return results


def to_markdown(results: list[MethodResult], n_questions: int, n_chunks: int) -> str:
    lines = [
        f"Questions: {n_questions}. Chunks in the index: {n_chunks}.",
        "",
        "| Method | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | ms/query |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        lines.append(
            f"| {r.method} | {r.recall_at_1:.3f} | {r.recall_at_5:.3f} | {r.recall_at_10:.3f} "
            f"| {r.mrr_at_10:.3f} | {r.ndcg_at_10:.3f} | {r.ms_per_query:.1f} |"
        )
    lines.append("")
    for r in results:
        lines.append(f"- {r.method} misses at 5: {', '.join(r.misses_at_5) or 'none'}")
    return "\n".join(lines) + "\n"
