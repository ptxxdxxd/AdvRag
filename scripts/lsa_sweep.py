"""Score the LSA and hybrid retrievers at several LSA sizes on the eval set."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from advrag.evaluate import evaluate, load_questions  # noqa: E402
from advrag.index import Index  # noqa: E402

questions = load_questions()
print("| LSA dims | LSA Recall@5 | LSA MRR@10 | Hybrid Recall@5 | Hybrid MRR@10 |")
print("| --- | --- | --- | --- | --- |")
for dims in (50, 100, 200, 300):
    lsa, hybrid = evaluate(Index.build(lsa_dims=dims), questions, methods=("lsa", "hybrid"))
    print(f"| {dims} | {lsa.recall_at_5:.3f} | {lsa.mrr_at_10:.3f} | {hybrid.recall_at_5:.3f} | {hybrid.mrr_at_10:.3f} |")
