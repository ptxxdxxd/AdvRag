"""Turn retrieved chunks into an answer that cites its sources.

Two answerers share one output shape:

* ``extractive_answer`` needs no API key. It quotes the sentences from the
  retrieved chunks that overlap the question most, each with its citation.
* ``claude_answer`` sends the chunks to Claude as documents with the API's
  citations feature turned on, so every cited span points back to a real chunk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .corpus import Chunk
from .text import tokenize

MODEL = "claude-opus-5-5"

SYSTEM_PROMPT = (
    "You answer questions about Python using only the documents provided, which are "
    "sections of Python Enhancement Proposals (PEPs). Keep the answer short. If the "
    "documents don't contain the answer, say that they don't, and don't answer from "
    "memory."
)


@dataclass
class Citation:
    chunk_id: str
    label: str
    quote: str


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    mode: str = "extractive"


_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z`\"(])")


def _clean(text: str) -> str:
    """Strip reST markup that reads badly in an answer: footnote refs, double backticks."""
    text = re.sub(r"\s*\[(\d+|#\w*)\]_", "", text)
    text = text.replace("``", "`")
    return re.sub(r"\s+", " ", text).strip()


def _sentences(text: str) -> list[str]:
    flat = _clean(text)
    return [s.strip() for s in _SENTENCE.split(flat) if len(s.split()) >= 4]


def extractive_answer(
    question: str,
    chunks: list[Chunk],
    max_sentences: int = 3,
    idf: dict[str, float] | None = None,
) -> Answer:
    """Pick the sentences whose words best match the question.

    A shared word counts by its IDF when one is given (rare words like ``strict``
    matter more than common ones like ``python``), with a small bonus for
    sentences from higher-ranked chunks.
    """
    q_terms = set(tokenize(question))
    weight = (lambda t: idf.get(t, 0.0)) if idf else (lambda t: 1.0)
    scored = []
    for rank, chunk in enumerate(chunks):
        for sent in _sentences(chunk.text):
            overlap = sum(weight(t) for t in q_terms & set(tokenize(sent)))
            if overlap > 0:
                score = overlap * (1.0 - 0.05 * rank)
                scored.append((-score, rank, len(sent), sent, chunk))
    scored.sort(key=lambda t: t[:3])
    if not scored:
        return Answer("The retrieved sections don't contain an answer to that.", [])
    picked, cites, used = [], [], set()
    for _, _, _, sent, chunk in scored[:max_sentences]:
        if chunk.id not in used:
            used.add(chunk.id)
            cites.append(Citation(chunk.id, chunk.label, sent))
        n = [c.chunk_id for c in cites].index(chunk.id) + 1
        picked.append(f"{sent} [{n}]")
    return Answer(" ".join(picked), cites, "extractive")


def build_request(question: str, chunks: list[Chunk]) -> dict:
    """The Messages API request: one plain-text document per chunk, citations on."""
    documents = [
        {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": chunk.text},
            "title": chunk.id,
            "context": chunk.label,
            "citations": {"enabled": True},
        }
        for chunk in chunks
    ]
    return {
        "model": MODEL,
        "max_tokens": 16000,
        "system": SYSTEM_PROMPT,
        "messages": [
            {"role": "user", "content": [*documents, {"type": "text", "text": question}]}
        ],
        "output_config": {"effort": "low"},
        # If a safety classifier declines, the API retries on a fallback model it picks.
        "betas": ["server-side-fallback-2026-07-01"],
        "fallbacks": "default",
    }


def parse_response(response, chunks: list[Chunk]) -> Answer:
    """Join text blocks and map each citation's document_index back to a chunk."""
    if getattr(response, "stop_reason", None) == "refusal":
        return Answer("The model declined to answer this question.", [], "claude")
    parts, cites, order = [], [], {}
    for block in response.content:
        if getattr(block, "type", None) != "text":
            continue
        parts.append(block.text)
        for cite in getattr(block, "citations", None) or []:
            idx = getattr(cite, "document_index", None)
            if idx is None or not 0 <= idx < len(chunks):
                continue  # never trust an index we didn't send
            chunk = chunks[idx]
            if chunk.id not in order:
                order[chunk.id] = len(order) + 1
                cites.append(Citation(chunk.id, chunk.label, cite.cited_text))
            parts.append(f" [{order[chunk.id]}]")
    return Answer("".join(parts).strip(), cites, "claude")


def claude_answer(question: str, chunks: list[Chunk], client=None) -> Answer:
    if client is None:
        import anthropic

        client = anthropic.Anthropic()
    response = client.beta.messages.create(**build_request(question, chunks))
    return parse_response(response, chunks)
