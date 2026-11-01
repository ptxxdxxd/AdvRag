import numpy as np

from advrag.evaluate import ndcg_at_k, recall_at_k, reciprocal_rank
from advrag.retrievers import BM25, LSA, Hybrid
from advrag.text import tokenize

DOCS = [
    "Use four spaces per indentation level.",
    "Limit all lines to a maximum of 79 characters.",
    "The zip function accepts strict to check lengths.",
    "Context managers define enter and exit methods.",
]


def test_tokenize_drops_stopwords_and_folds_plurals():
    assert tokenize("The spaces of the lines") == ["space", "line"]


def test_bm25_ranks_the_matching_doc_first():
    bm25 = BM25(DOCS)
    assert bm25.search("how many characters per line", k=1)[0][0] == 1
    assert bm25.search("zip strict", k=1)[0][0] == 2


def test_bm25_unknown_terms_score_zero():
    assert not np.any(BM25(DOCS).scores("quantum chromodynamics"))


def test_lsa_returns_unit_vectors():
    lsa = LSA(DOCS * 3, dims=4, min_df=1)
    norms = np.linalg.norm(lsa.doc_vecs, axis=1)
    assert np.allclose(norms, 1.0)


def test_hybrid_fuses_ranks():
    class Fixed:
        def __init__(self, order):
            self.order = order

        def search(self, query, k=10):
            return [(i, 1.0) for i in self.order[:k]]

    fused = Hybrid([Fixed([0, 1]), Fixed([1, 2])]).search("q", k=3)
    # 1 shows up in both lists, so it beats 0 and 2, which each appear once.
    assert [i for i, _ in fused] == [1, 0, 2]


def test_metrics():
    ranked = ["a", "b", "c"]
    assert recall_at_k(ranked, {"c"}, 2) == 0.0
    assert recall_at_k(ranked, {"c"}, 3) == 1.0
    assert reciprocal_rank(ranked, {"b"}) == 0.5
    assert ndcg_at_k(ranked, {"a"}) == 1.0
    assert 0 < ndcg_at_k(ranked, {"b"}) < 1
