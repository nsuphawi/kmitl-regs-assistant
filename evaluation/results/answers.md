# Answer evaluation

2026-11-04 23:51 · model `gemma-4-E4B-it` · 36 items · MIN_SCORE=0.4 · judge = same model (see limitations)

| system | accuracy (answerable) | abstain acc (unanswerable) | cite precision | valid w/o repair | avg steps | avg latency (s) |
|---|---|---|---|---|---|---|
| closed | 0.000 | 1.000 | 0.000 | 1.000 | 0.00 | 0.64 |
| rag | 0.817 | 1.000 | 0.575 | 0.972 | 1.00 | 4.71 |
| agent | 0.750 | 0.833 | 0.583 | 1.000 | 1.89 | 4.93 |

Per-item outputs (use these to pick failure cases): `evaluation/results/answers_items.jsonl`
