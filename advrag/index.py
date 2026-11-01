"""Build every retriever over one corpus."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .corpus import Chunk, load_corpus
from .retrievers import BM25, LSA, Hybrid, Retriever

DEFAULT_CORPUS = Path(__file__).resolve().parent.parent / "data" / "peps"


@dataclass
class Index:
    chunks: list[Chunk]
    retrievers: dict[str, Retriever] = field(default_factory=dict)

    @classmethod
    def build(cls, corpus_dir: Path | str = DEFAULT_CORPUS, lsa_dims: int = 100) -> "Index":
        chunks = load_corpus(corpus_dir)
        texts = [f"{c.pep_title}\n{c.section}\n{c.text}" for c in chunks]
        bm25 = BM25(texts)
        lsa = LSA(texts, dims=lsa_dims)
        return cls(chunks, {"bm25": bm25, "lsa": lsa, "hybrid": Hybrid([bm25, lsa])})

    def search(self, query: str, method: str = "hybrid", k: int = 5) -> list[tuple[Chunk, float]]:
        return [(self.chunks[i], s) for i, s in self.retrievers[method].search(query, k)]
