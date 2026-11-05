Questions: 42. Chunks in the index: 670.

| Method | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | ms/query |
| --- | --- | --- | --- | --- | --- | --- |
| bm25 | 0.405 | 0.786 | 0.857 | 0.547 | 0.576 | 0.5 |
| lsa | 0.357 | 0.810 | 0.905 | 0.537 | 0.586 | 0.3 |
| hybrid | 0.405 | 0.857 | 0.881 | 0.590 | 0.625 | 0.9 |

- bm25 misses at 5: q03, q05, q09, q19, q21, q23, q32, q37, q40
- lsa misses at 5: q03, q05, q11, q21, q22, q23, q32, q37
- hybrid misses at 5: q03, q05, q19, q23, q32, q37
