# KMITL Regulations Assistant

Ask questions about KMITL's official student regulations and guidebooks and get a cited answer, or a clear "not in the documents". An agent decides whether to search the documents, use a calculator (for GPA questions) or answer directly. Every answer is schema-validated, and every citation is checked against the passages the model was actually shown.

**Who it's for:** KMITL students, especially international students, who need answers like "how many credits can I register?", "what GPA gets me dismissed?" or "how long can I take leave?". Today those answers are buried in five long PDFs.

**Corpus:** five public PDFs linked from the KMITL Office of International Affairs ([oia.kmitl.ac.th/guidebook-regulation](https://oia.kmitl.ac.th/guidebook-regulation/)): the undergraduate and graduate regulations (2016), the undergraduate and graduate guidebooks (2022), and the general guidebook. The course reference app cannot answer questions about these. See [`data/sources.json`](data/sources.json).

## Course capability families covered

| # | Family | Where |
|---|---|---|
| 1 | Structured output + schema validation (one repair retry, citation allow-check) | `app/schemas.py`, `app/answer.py` |
| 2 | RAG: clause-aware chunking (split at Article/clause numbers, Article heading prefixed) → embed → index → hybrid retrieve (BM25 + vector, RRF, optional HyDE expansion) → cite | `app/ingest.py`, `app/store.py`, `app/retrieve.py` |
| 3 | **Retrieval evaluation**: hit@k, recall@k, MRR on a labelled set; baselines; chunk-size and `min_score` sweeps | `evaluation/` |
| 4 | **Agent loop with tools**: native function calling with a prompted-JSON fallback, allow-list, step cap, repeat-query guard | `app/agent.py`, `app/tools.py` |
| 6 | Served interface: FastAPI `POST /ask` and an SSE stream of agent steps with a small web UI | `app/api.py`, `app/static/index.html` |

## Architecture

```mermaid
flowchart LR
    U[User / Web UI / CLI] -->|POST /ask, GET /ask/stream SSE| API[FastAPI]
    API --> AG[Agent loop<br/>max_steps, repeat guard]
    AG -->|tool call| V{allow-list +<br/>arg validation}
    V -->|search_docs| R[Hybrid retriever]
    V -->|calculator| C[AST-only calculator]
    R --> B[BM25] & E[Vector cosine]
    B & E --> F[RRF fuse k=60] --> M[min_score floor] --> AG
    AG -->|final JSON| S[Pydantic validate<br/>+ citation check<br/>+ 1 repair]
    S --> API
    AG <-->|temperature=0| L[(Class proxy<br/>gemma-4-E4B-it)]
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # fill in OPENAI_API_KEY / OPENAI_BASE_URL (never commit .env)

python -m scripts.fetch_corpus  # downloads the 5 PDFs into data/raw/ and writes MANIFEST.json
python -m scripts.ocr_scanned   # OCR the 2 image-only PDFs -> .txt (already committed; needs tesseract)
python -m scripts.ocr_clean     # strip translation-agency stamps from the OCR text (idempotent)
python -m scripts.build_index   # chunk + embed -> data/index/
```

If Google Drive refuses the scripted download, open the `page` link in `data/sources.json`, save the PDF as `data/raw/<name>.pdf`, and rerun `fetch_corpus` to refresh the manifest. Commit the PDFs so the numbers can be reproduced.

**Embeddings:** the default `EMBED_PROVIDER=local` runs `all-MiniLM-L6-v2` on your machine. If the class proxy serves an embedding model, set `EMBED_PROVIDER=openai` and `EMBED_MODEL=<name>`. `tfidf` is an offline fallback that the unit tests use.

## Run

```bash
python -m scripts.ask "What is the maximum number of credits per regular semester?"   # agent
python -m scripts.ask "..." --mode rag --show-context                                  # single-shot RAG
python -m scripts.ask "..." --mode closed                                             # no-retrieval baseline
uvicorn app.api:app --reload    # open http://127.0.0.1:8000
```

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "How many semesters of leave can I take?", "mode": "agent"}'
```

## Evaluation (reproducible)

**1. Build the labelled set (about 30 items).** Gold labels are verbatim **evidence quotes**, not chunk ids, so the same labels work at every chunk size.

```bash
python -m scripts.draft_eval --n 40      # model drafts Q/A/evidence; the quote is checked to be verbatim
# hand-review data/eval/qa_draft.jsonl: fix wording (paraphrase, don't copy the PDF), check answers,
# set "verified": true, then:
cat data/eval/qa_draft.jsonl >> data/eval/qa.jsonl
# review the out-of-scope questions too, confirm the PDFs really don't answer them, then:
cat data/eval/unanswerable_seed.jsonl >> data/eval/qa.jsonl
```

Aim for about 24 answerable items (mix factual, numeric, keyword like form codes, and paraphrased) plus about 6 unanswerable ones. Add a few questions you wrote yourself as a student; model-drafted questions are easier than real ones.

**2. Retrieval: baselines, chunk size and `min_score`.** No LLM calls.

```bash
python -m evaluation.run_retrieval_eval --strategies window,clause --chunk-sizes 100,150,300
python -m evaluation.run_retrieval_eval --hyde        # + HyDE query expansion (1 model call per question)
python -m evaluation.run_retrieval_eval --rerank      # optional: + LLM 1-10 reranker
```

This writes `evaluation/results/retrieval.md`:

- **Baselines:** BM25-only and vector-only, compared with hybrid RRF, on hit@1/3/5, recall@5 and MRR.
- **Measured improvement over a naive default:** fixed-window vs clause-aware chunking, the chunk-size sweep, HyDE on/off, and the `min_score` sweep. The `min_score` sweep trades answerable hit@5 against correct abstention on unanswerable questions. Put the winning values in `.env` and say why in the report.

**3. End-to-end answers.** Closed-book baseline vs RAG vs agent.

```bash
python -m evaluation.run_answer_eval
```

This writes `evaluation/results/answers.md` (accuracy judged by an LLM against your reference answers, abstention accuracy, citation precision, schema validity, steps, latency) and `answers_items.jsonl`, the per-item outputs you'll use to pick failure cases for the report and demo.

## Safety and correctness

- Config comes only from the environment (`.env` is gitignored). The model runs at `temperature=0` everywhere.
- The model only *requests* tools. `app/tools.py` enforces an allow-list, validates and clamps arguments (`k` 1–8, query ≤300 chars), and the calculator walks a whitelisted AST (no `eval`).
- Tool output is wrapped as untrusted data, and the system prompt tells the model to ignore any instructions inside retrieved text.
- Final answers are validated with Pydantic. Citations to chunks the model never saw are dropped, and an answer with no valid citation gets one repair attempt, then abstains.
- If nothing clears `min_score`, the RAG path abstains without calling the model.

## Tests

```bash
pytest -q     # 25 offline tests: metrics, RRF, BM25 rare tokens, min_score, tools, agent loop (fake LLM)
```

## Layout

```
app/          config, llm client, ingest/chunk, embeddings, bm25, store, retrieve, schemas, answer, tools, agent, api
scripts/      fetch_corpus, build_index, draft_eval, ask
evaluation/   metrics, dataset (evidence-quote gold), run_retrieval_eval, run_answer_eval, results/
data/         sources.json, raw/ (PDFs + MANIFEST), eval/ (qa.jsonl)
tests/        offline unit tests with a fictional fixture corpus
```

## Known limitations

- Two PDFs (the 2022 undergraduate and graduate regulations, filed on the OIA page as "guidebooks") are scans with no text layer, so they are OCR'd with Tesseract. OCR errors (e.g. `B-E.` for `B.E.`) can hurt BM25 matching.
- The corpus holds both the 2016 and the 2022 regulations, which can disagree. The answer should cite the 2022 version when both match.
- PDF text extraction loses table structure, so grade tables and fee tables may chunk badly.
- The LLM judge is the same small model as the generator. Spot-check its verdicts by hand and report the agreement rate.
- The corpus covers English documents only. The 2016 regulations may have been amended since.

## Error analysis tool

```bash
python -m scripts.debug_rank "How many credits can an undergraduate register in a regular semester?" --find "not more than 22 credits"
```

Shows where the chunk containing the answer ranks in each list (BM25 / vector, question / HyDE), after RRF fusion, and whether `MIN_SCORE` removes it.

Worked example: with 300-word fixed windows the credit-limit clause (2022 regulations, Art. 14.4) sat at the end of a chunk about tuition fees and ranked 39th (BM25) and 109th (vector), so it never reached the model and the app answered "not specified". Clause-aware chunking isolates 14.4 under its Article heading.
