# AdvRag

Question answering over Python's design documents (PEPs), with retrieval you can measure and answers that cite their sources.

Ask a question, AdvRag finds the PEP sections most likely to answer it, then either quotes the best sentences or asks Claude to answer from those sections only. Every answer lists the exact sections it used. A hand-labeled set of 42 questions scores three retrievers, so changes are judged by numbers instead of by feel.

```text
$ advrag ask "What configure option builds CPython without the GIL?"
A new build configuration flag, `--disable-gil` will be added to the configure script that will
build CPython with support for running without the global interpreter lock. [1] ...

Sources:
  [1] PEP 703: Making the Global Interpreter Lock Optional in CPython > Build Configuration Changes (pep-0703#build-configuration-changes)
  [2] PEP 703: Making the Global Interpreter Lock Optional in CPython > Abstract (pep-0703#abstract)
```

## How it works

1. **Corpus.** 24 PEPs (PEP 8, 20, 257, 343, 405, 484, 498, 518, 526, 557, 572, 584, 585, 604, 618, 634, 636, 657, 668, 695, 703, 723, 3107, 3333), committed under `data/peps/`. All are public domain or CC0. `data/peps/MANIFEST.json` records each file's SHA-256 and `scripts/fetch_corpus.py` checks the copies against it or against upstream.
2. **Chunking.** Each PEP is split on its section headings, so a chunk is one section, with long sections cut into parts of about 400 words. That gives 670 chunks with readable ids like `pep-0008#maximum-line-length`. Copyright, references and footnotes sections are dropped.
3. **Retrieval.** Three retrievers, all in plain NumPy, so nothing needs a GPU or a model download:
   - **BM25**: the standard lexical ranking function.
   - **LSA**: TF-IDF vectors reduced to 100 dimensions with a truncated SVD. It can match a question to a section that shares no exact words with it.
   - **Hybrid**: reciprocal rank fusion of the two.
4. **Answering.**
   - Default: an extractive answer. It quotes the sentences from the top chunks that best match the question, weighted by how rare each shared word is, and cites each one. It says so when nothing matches.
   - `--llm`: sends the top chunks to Claude as separate documents with the API's citations feature turned on. Each citation comes back as a document index, which is mapped to the chunk it came from. An index that wasn't sent is ignored.

## Results

From `advrag eval` on the 42 questions in `evals/questions.jsonl`. Recall@k is the share of questions with a correct section in the top k.

| Method | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
| --- | --- | --- | --- | --- | --- |
| BM25 | 0.405 | 0.786 | 0.857 | 0.547 | 0.576 |
| LSA | 0.357 | 0.810 | 0.905 | 0.537 | 0.586 |
| Hybrid | 0.405 | 0.857 | 0.881 | 0.590 | 0.625 |

Hybrid finds a correct section in the top 5 for 36 of 42 questions, against 33 for BM25 and 34 for LSA. Retrieval took about a millisecond per query on the machine it was tested on. The full output, including which questions each method missed, is in `results/retrieval.md`.

What the numbers don't show yet:

- **The set is small.** One question moves recall by 0.024, so a gap of one or two questions between methods is within noise.
- **The LSA size was picked on this same set.** `results/lsa_sweep.md` has the sweep. 100 dimensions did best for hybrid recall and MRR, which flatters the hybrid row a little. A held-out split would fix that.
- **I wrote the questions and labels.** Each question names its gold sections and a phrase that must appear in one of them, and a test checks every label against the corpus. After the first run I added one gold section (q12), because `pep-0484#storing-and-distributing-stub-files` answers that question as well as the section first labeled. No other label changed.
- **The misses are real.** Hybrid still misses q03, q05, q19, q23, q32 and q37. q05 ("Is it OK to check for None with ==?") is a typical failure: once stopwords and the `==` symbol are gone, almost nothing of the question is left to match.
- **Answer quality isn't scored.** The eval measures retrieval only. The extractive answers are rough: they quote whole sentences, so they can include text around the answer.

## Run it

Python 3.10 or newer.

```bash
pip install -e .            # NumPy is the only requirement; -e because the corpus is read from data/
advrag eval                 # score the three retrievers
advrag search "dictionary merge operator" -k 3
advrag ask "How do I make zip complain when inputs have different lengths?"
```

To answer with Claude:

```bash
pip install -e ".[llm]"
export ANTHROPIC_API_KEY=...
advrag ask --llm "Why can't I write y := f(x) on its own line?"
```

Tests:

```bash
pip install -e ".[dev]"
pytest
```

## Not verified

- **The `--llm` path has not been run against the real Claude API.** The tests send the request to a local fake server, which confirms the SDK accepts the request and that citations are mapped back to chunks. They don't confirm the API's answers or citations. The request uses `claude-opus-5-5` at low effort, with server-side refusal fallback turned on.
- The old version of this repo, with fine-tuning, a Streamlit app and monitoring, is kept in `legacy/` for reference. It wasn't run or tested as part of this rebuild and isn't maintained.

## Layout

```text
advrag/        corpus loading, retrievers, answerers, metrics, CLI
data/peps/     the 24 PEPs and their checksums
evals/         questions.jsonl: 42 questions with gold sections
results/       eval output and the LSA size sweep
scripts/       corpus verification and the LSA sweep
tests/         unit tests, plus checks that every eval label is valid
legacy/        the previous version, unmaintained
```

## Next steps

- Add a pretrained sentence-embedding retriever next to LSA and compare it on the same questions.
- Split the questions into a tuning set and a held-out test set.
- Score answers as well as retrieval: whether the answer is right and whether every claim is cited.
