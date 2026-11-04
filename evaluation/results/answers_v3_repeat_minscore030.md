# Answer evaluation

2026-11-04 22:29 · model `gemma-4-E4B-it` · 36 items · MIN_SCORE=0.3 · judge = same model (see limitations)

| system | accuracy (answerable) | abstain acc (unanswerable) | cite precision | valid w/o repair | avg steps | avg latency (s) |
|---|---|---|---|---|---|---|
| closed | 0.000 | 1.000 | 0.000 | 1.000 | 0.00 | 0.87 |
| rag | 0.800 | 1.000 | 0.557 | 0.972 | 1.00 | 4.48 |
| agent | 0.783 | 0.833 | 0.583 | 1.000 | 1.86 | 4.77 |

Per-item outputs (use these to pick failure cases): `evaluation/results/answers_items.jsonl`
