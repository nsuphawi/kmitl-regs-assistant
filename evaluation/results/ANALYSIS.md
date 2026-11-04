# Evaluation analysis (working notes for the report)

Eval set: 36 items in `data/eval/qa.jsonl`. 30 are answerable: 29 were drafted by the model and then reviewed by hand, and c01 is a real failure case found while building the app. The other 6 are unanswerable, and a text search confirmed they are absent from the corpus.
Model: `gemma-4-E4B-it` at temperature 0. Embeddings: `all-MiniLM-L6-v2` (local).
With n = 30, one question is 3.3 points, so a difference of 1–2 questions is within noise.

## 1. Retrieval: chunking strategy (the measured improvement over the naive default)

| chunking | words | mode | hit@5 | recall@5 | MRR |
|---|---|---|---|---|---|
| fixed window (naive) | 150 | vector | 0.733 | 0.501 | 0.531 |
| **clause-aware** | 150 | vector | **0.800** | **0.585** | **0.551** |
| fixed window (naive) | 150 | hybrid | 0.733 | 0.515 | 0.465 |
| **clause-aware** | 150 | hybrid | **0.767** | **0.550** | 0.456 |
| clause-aware | 150 | hybrid + HyDE | 0.767 | 0.593 | 0.551 |
| clause-aware | 150 | BM25 only | 0.567 | 0.431 | 0.411 |

Source: `retrieval_v1.md` (full grid: window/clause × 100/150/300 words × bm25/vector/hybrid).

- Clause-aware chunking splits at Article/clause numbers and puts the Article heading in front of each chunk. It beats fixed windows at the same size: vector recall@5 goes from 0.501 to 0.585 (+0.08).
- Motivating failure (c01, "how many credits per regular semester"): with 300-word windows, clause 14.4 sat at the end of a chunk about tuition fees. It ranked 39th (BM25) and 109th (vector), never reached the model, and the app answered "not specified". With clause chunks it is retrieved and answered correctly.
- BM25 is weak here. The questions are paraphrased, and the 2016 PDFs extract with broken words ("tha n", "re gular"). As a result, hybrid fusion does not beat vector alone on hit@5. Honest conclusion: hybrid + HyDE ties vector on MRR and is best on recall@5, but the gap is within noise.

## 2. min_score floor

| min_score | answerable hit@5 | unanswerable abstained before the LLM |
|---|---|---|
| 0.30 | 0.767 | 0.00 |
| 0.35 | 0.767 | 0.33 |
| **0.40** | **0.767** | **0.50** |
| 0.45 | 0.733 | 0.50 |
| 0.50 | 0.733 | 0.67 |
| 0.55 | 0.633 | 0.83 |

The "balanced" score peaks at 0.55, but that costs 4 answerable questions. The RAG model already abstains on 6/6 unanswerable questions by itself, even at 0.30. So the floor's job is to save model calls and add a safety margin, not to do the abstaining. **Chosen value: 0.40**, the highest floor with zero loss on answerable questions.

## 3. End-to-end answers

| system | accuracy (answerable) | abstain acc. | schema valid w/o repair |
|---|---|---|---|
| closed-book baseline (no retrieval) | 0.000 | 1.000 | 1.000 |
| RAG v1 | 0.533 | 1.000 | 0.722 |
| agent v1 | 0.483 | 0.833 | 0.750 |
| **RAG v2** | **0.800** | 1.000 | 0.972 |
| **agent v2** | **0.717** | 0.833 | 1.000 |

Closed-book: the model refused 29 of 30 times. It does not know KMITL's rules, so retrieval is necessary.

### What changed from v1 to v2, and how much each change contributed (RAG, +8.0 questions net)

| cause | items | effect |
|---|---|---|
| Validation fix: lenient JSON (raw newlines), cleaned `[id]` citations, truncated long quotes, inline-citation salvage | d03, d10, d13, d16, d23, c01 (+1 each); d17, d32, d35 (+0.5 each) | **+7.5** |
| Judge rubric changed *after seeing v1*: extra non-contradicting detail no longer lowers the grade | d20, d31, d33, d37 (+0.5 each) | +2.0 |
| Run-to-run variance (temperature 0, but HyDE and the answer differ between runs) | d02 (−1), d19 (−0.5) | −1.5 |

In v1, 9 answerable questions were wrongly turned into "I could not find this" because the model's JSON failed validation twice. In 4 of them the right chunk *had* been retrieved. That was the largest single source of error, and it came from generation and validation, not retrieval. The judge change is disclosed here because it was made after looking at the results. Its verdicts should be spot-checked by hand.

### Repeatability: two identical runs (v2 vs v3, same code, MIN_SCORE=0.30)

| system | accuracy v2 | accuracy v3 | items whose score changed |
|---|---|---|---|
| RAG | 0.800 | 0.800 | 0 / 36 |
| agent | 0.717 | 0.783 | 5 / 36 (d02, d10, d17, d38, c01) |

RAG is fully repeatable at temperature 0. The agent is not, even at temperature 0, because it writes its own search query and decides when to stop, so small differences compound over steps. A single agent number therefore carries about ±0.07 of run-to-run noise. This is why the RAG pipeline is reported as the main system and the agent as the interactive front-end.

## 4. Remaining failure cases (v2)

- **Retrieval misses (gold not in the top 5):** d17, d19, d26, d32, d35. Most come from the OCR'd 2022 graduate regulations, or the question wording differs from the regulation's (d26: "bypass examinations" vs "skip the exams of related courses").
- **Wrong document level:** agent d02 answered with the graduate rule for an undergraduate question. Both levels have near-identical clauses.
- **Reference answer too narrow (label issue, not a model error):** d39 (72 vs 48 credits depending on the plan) and d38. The model's fuller answer was graded "partial".
- **Abstained despite retrieval:** agent c01, RAG d02. The flagship question c01 passes in RAG but fails in agent mode in this run, a reminder that one run is not proof.
- **Scoring artifact:** agent u06 refused in prose ("I do not have access to ...") but did not set `abstained: true`, so it counts as a failed abstention.

## 5. Limitations

- The judge is the same small model as the generator, and its rubric was changed after v1.
- n = 30, and model-drafted questions are easier than real student questions.
- Some evidence comes from Tesseract OCR text. The 2016 and 2022 regulations overlap and sometimes differ.
