# Answer evaluation

2026-11-04 17:15 · model `gemma-4-E4B-it` · 36 items · MIN_SCORE=0.3 · judge = same model (see limitations)

| system | accuracy (answerable) | abstain acc (unanswerable) | cite precision | valid w/o repair | avg steps | avg latency (s) |
|---|---|---|---|---|---|---|
| closed | 0.000 | 1.000 | 0.000 | 1.000 | 0.00 | 0.58 |
| rag | 0.533 | 1.000 | 0.603 | 0.722 | 1.00 | 7.43 |
| agent | 0.483 | 0.833 | 0.530 | 0.750 | 2.11 | 6.62 |

Per-item outputs (use these to pick failure cases): `evaluation/results/answers_items.jsonl`
