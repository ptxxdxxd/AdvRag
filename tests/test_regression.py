"""Retrieval quality gate.

Floors sit one question below the scores in results/retrieval.md, so a change
that loses two or more questions at Recall@5 fails CI. Raise them when a
change improves the numbers; lower them only with a written reason.
"""

import pytest

from advrag.evaluate import evaluate, load_questions
from advrag.index import Index

FLOORS_AT_5 = {"bm25": 32, "lsa": 33, "hybrid": 35}  # out of 42 questions


@pytest.fixture(scope="module")
def results():
    questions = load_questions()
    index = Index.build()
    return {r.method: r for r in evaluate(index, questions)}, len(questions)


@pytest.mark.parametrize("method", sorted(FLOORS_AT_5))
def test_recall_at_5_does_not_regress(results, method):
    by_method, n = results
    hits = round(by_method[method].recall_at_5 * n)
    assert hits >= FLOORS_AT_5[method], (
        f"{method}: {hits}/{n} at Recall@5, floor is {FLOORS_AT_5[method]}; "
        f"misses: {', '.join(by_method[method].misses_at_5)}"
    )


def test_hybrid_beats_each_retriever_alone_on_mrr(results):
    by_method, _ = results
    assert by_method["hybrid"].mrr_at_10 >= by_method["bm25"].mrr_at_10
    assert by_method["hybrid"].mrr_at_10 >= by_method["lsa"].mrr_at_10
