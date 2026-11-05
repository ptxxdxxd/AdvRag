"""Three retrievers over the same chunks: BM25 (lexical), LSA (dense), and a hybrid.

All of them return ``(chunk_index, score)`` pairs, best first. Only NumPy is
needed, so the whole pipeline runs offline with no model downloads.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Protocol, Sequence

import numpy as np

from .text import tokenize


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]: ...


def _top_k(scores: np.ndarray, k: int) -> list[tuple[int, float]]:
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx], kind="stable")]
    return [(int(i), float(scores[i])) for i in idx]


class BM25:
    """Okapi BM25 with the usual k1/b parameters."""

    name = "bm25"

    def __init__(self, texts: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lengths = np.array([sum(d.values()) for d in self.docs], dtype=float)
        self.avgdl = float(self.lengths.mean()) if len(self.docs) else 0.0
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log((n - f + 0.5) / (f + 0.5) + 1.0) for t, f in df.items()}
        # Inverted index: term -> list of (doc, term frequency)
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for i, d in enumerate(self.docs):
            for t, tf in d.items():
                self.postings.setdefault(t, []).append((i, tf))

    def scores(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.docs))
        norm = self.k1 * (1 - self.b + self.b * self.lengths / (self.avgdl or 1.0))
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in self.postings[term]:
                scores[i] += idf * tf * (self.k1 + 1) / (tf + norm[i])
        return scores

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        return _top_k(self.scores(query), k)


class LSA:
    """Latent semantic analysis: TF-IDF vectors reduced with a truncated SVD.

    It's a classic dense retriever that needs no pretrained model. It can match a
    query to a chunk that shares no exact words with it, which BM25 can't.
    """

    name = "lsa"

    def __init__(self, texts: Sequence[str], dims: int = 200, min_df: int = 2):
        docs = [Counter(tokenize(t)) for t in texts]
        df: Counter[str] = Counter()
        for d in docs:
            df.update(d.keys())
        self.vocab = {t: i for i, t in enumerate(sorted(t for t, f in df.items() if f >= min_df))}
        n = len(docs)
        self.idf = np.zeros(len(self.vocab))
        for t, i in self.vocab.items():
            self.idf[i] = math.log((1 + n) / (1 + df[t])) + 1.0
        matrix = np.zeros((n, len(self.vocab)))
        for row, d in enumerate(docs):
            for t, tf in d.items():
                col = self.vocab.get(t)
                if col is not None:
                    matrix[row, col] = 1.0 + math.log(tf)
        matrix *= self.idf
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True).clip(min=1e-12)
        u, s, vt = np.linalg.svd(matrix, full_matrices=False)
        self.dims = min(dims, len(s))
        self.components = vt[: self.dims]  # dims x vocab
        doc_vecs = u[:, : self.dims] * s[: self.dims]
        self.doc_vecs = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True).clip(min=1e-12)

    def embed(self, query: str) -> np.ndarray:
        vec = np.zeros(len(self.vocab))
        for t, tf in Counter(tokenize(query)).items():
            col = self.vocab.get(t)
            if col is not None:
                vec[col] = 1.0 + math.log(tf)
        vec *= self.idf
        q = self.components @ vec
        norm = np.linalg.norm(q)
        return q / norm if norm > 0 else q

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        return _top_k(self.doc_vecs @ self.embed(query), k)


class Hybrid:
    """Reciprocal rank fusion of several retrievers (Cormack et al., 2009)."""

    name = "hybrid"

    def __init__(self, retrievers: Sequence[Retriever], k_rrf: int = 60, depth: int = 50):
        self.retrievers, self.k_rrf, self.depth = list(retrievers), k_rrf, depth

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        fused: dict[int, float] = {}
        for r in self.retrievers:
            for rank, (i, _) in enumerate(r.search(query, self.depth), start=1):
                fused[i] = fused.get(i, 0.0) + 1.0 / (self.k_rrf + rank)
        ranked = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:k]
