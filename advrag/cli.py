"""Command line: ``advrag ask``, ``advrag search`` and ``advrag eval``."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .answer import claude_answer, extractive_answer
from .evaluate import evaluate, load_questions, to_markdown
from .index import Index


def _print_answer(answer) -> None:
    print(answer.text)
    if answer.citations:
        print("\nSources:")
        for n, c in enumerate(answer.citations, start=1):
            print(f"  [{n}] {c.label} ({c.chunk_id})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="advrag", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    ask = sub.add_parser("ask", help="answer a question with citations")
    ask.add_argument("question")
    ask.add_argument("--method", choices=["bm25", "lsa", "hybrid"], default="hybrid")
    ask.add_argument("-k", type=int, default=5, help="chunks to retrieve")
    ask.add_argument(
        "--llm", action="store_true",
        help="answer with Claude (needs Anthropic credentials) instead of quoting sentences",
    )

    search = sub.add_parser("search", help="show the top chunks for a query")
    search.add_argument("query")
    search.add_argument("--method", choices=["bm25", "lsa", "hybrid"], default="hybrid")
    search.add_argument("-k", type=int, default=5)

    ev = sub.add_parser("eval", help="score every retriever on the labeled questions")
    ev.add_argument("--out", type=Path, help="also write the results table to this file")

    args = parser.parse_args(argv)
    index = Index.build()

    if args.cmd == "search":
        for chunk, score in index.search(args.query, args.method, args.k):
            print(f"{score:8.4f}  {chunk.id}  ({chunk.label})")
        return 0

    if args.cmd == "ask":
        chunks = [c for c, _ in index.search(args.question, args.method, args.k)]
        if args.llm:
            _print_answer(claude_answer(args.question, chunks))
        else:
            _print_answer(extractive_answer(args.question, chunks, idf=index.retrievers["bm25"].idf))
        return 0

    questions = load_questions()
    table = to_markdown(evaluate(index, questions), len(questions), len(index.chunks))
    print(table)
    if args.out:
        args.out.write_text(table, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
