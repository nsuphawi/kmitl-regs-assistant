# Retrieval evaluation

2026-11-04 21:33 · embeddings: `local` · 30 answerable / 6 unanswerable questions

| chunking | words | mode | hit@1 | hit@3 | hit@5 | recall@5 | MRR |
|---|---|---|---|---|---|---|---|
| clause | 150 | bm25 | 0.300 | 0.500 | 0.567 | 0.431 | 0.411 |
| clause | 150 | vector | 0.433 | 0.600 | 0.800 | 0.585 | 0.551 |
| clause | 150 | hybrid | 0.267 | 0.633 | 0.767 | 0.550 | 0.456 |

## min_score sweep (hybrid, clause 150-word chunks)

| min_score | answerable hit@5 | unanswerable correctly abstained | balanced |
|---|---|---|---|
| 0.0 | 0.767 | 0.000 | 0.383 |
| 0.2 | 0.767 | 0.000 | 0.383 |
| 0.3 | 0.767 | 0.000 | 0.383 |
| 0.35 | 0.767 | 0.333 | 0.550 |
| 0.4 | 0.767 | 0.500 | 0.633 |
| 0.45 | 0.733 | 0.500 | 0.617 |
| 0.5 | 0.733 | 0.667 | 0.700 |
| 0.55 | 0.633 | 0.833 | 0.733 |
| 0.6 | 0.567 | 0.833 | 0.700 |

Best balanced threshold: **0.55** — set `MIN_SCORE=0.55` in `.env`.
