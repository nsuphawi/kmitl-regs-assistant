"""End-to-end answer evaluation: closed-book baseline vs RAG vs agent.

    python -m evaluation.run_answer_eval                       # all systems
    python -m evaluation.run_answer_eval --systems closed,rag  # subset

Metrics
  accuracy       LLM-as-judge vs the reference answer (correct=1, partial=0.5) on answerable items
  abstain_acc    unanswerable items where the system abstained (deterministic, no judge)
  cite_precision share of cited chunks that contain the gold evidence (answerable, non-abstained)
  validity       share of outputs that passed schema validation without a repair
Writes evaluation/results/answers.md and answers_items.jsonl (for failure analysis).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from app.agent import run_agent
from app.answer import answer_closed_book, answer_rag
from app.config import ROOT, settings
from app.llm import chat, extract_json
from app.retrieve import Retriever
from app.store import Index

from .dataset import load_eval, relevant_ids
from .metrics import mean

RESULTS = ROOT / "evaluation" / "results"

JUDGE = (
    "You grade answers about KMITL regulations. Compare the CANDIDATE to the REFERENCE answer. "
    "'correct' = contains the reference's key facts (numbers, conditions) with no contradiction. "
    "Extra details that do not contradict the reference (e.g. exceptions, the graduate-level rule, "
    "article numbers) do NOT lower the grade. 'partial' = some key facts right but a key fact missing; "
    "'incorrect' = wrong, contradicting, or a refusal. "
    'Reply with JSON only: {"verdict": "correct|partial|incorrect", "reason": "<one sentence>"}'
)


def judge(question: str, reference: str, candidate: str) -> tuple[float, str]:
    msg = [{"role": "system", "content": JUDGE},
           {"role": "user", "content": f"QUESTION: {question}\nREFERENCE: {reference}\nCANDIDATE: {candidate}"}]
    try:
        out = extract_json(chat(msg).choices[0].message.content)
        verdict = str(out.get("verdict", "incorrect")).lower()
        return {"correct": 1.0, "partial": 0.5}.get(verdict, 0.0), str(out.get("reason", ""))[:200]
    except (ValueError, AttributeError):
        return 0.0, "judge output unparseable"


def run_system(name: str, it: dict, r: Retriever) -> dict:
    t0 = time.time()
    if name == "closed":
        res = answer_closed_book(it["question"])
        ans, repaired, steps, error = res.answer, res.repaired, 0, res.error
    elif name == "rag":
        res = answer_rag(it["question"], r)
        ans, repaired, steps, error = res.answer, res.repaired, 1, res.error
    elif name == "agent":
        res = run_agent(it["question"], r)
        repairs = [e["error"] for e in res.trace if e["type"] == "repair"]
        ans, repaired, steps, error = res.answer, bool(repairs), res.steps, (repairs[0] if repairs else None)
    else:
        raise ValueError(name)
    return {"answer": ans.answer, "abstained": ans.abstained, "cited": [c.chunk_id for c in ans.citations],
            "repaired": repaired, "validation_error": error, "steps": steps, "latency_s": round(time.time() - t0, 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", type=Path, default=ROOT / "data" / "eval" / "qa.jsonl")
    ap.add_argument("--systems", default="closed,rag,agent")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--include-unverified", action="store_true")
    args = ap.parse_args()

    items = load_eval(args.qa, args.include_unverified)
    if args.limit:
        items = items[: args.limit]
    r = Retriever(Index.load(settings.index_dir))
    gold = {it["id"]: relevant_ids(it, r.chunks) for it in items}
    systems = args.systems.split(",")
    RESULTS.mkdir(parents=True, exist_ok=True)

    per_item = []
    for it in items:
        for sysname in systems:
            out = run_system(sysname, it, r)
            if it["answerable"]:
                score, reason = (0.0, "abstained") if out["abstained"] else judge(it["question"], it["answer"], out["answer"])
            else:
                score, reason = (1.0 if out["abstained"] else 0.0), "abstention check"
            cites_ok = [c in gold[it["id"]] for c in out["cited"]]
            per_item.append({"id": it["id"], "system": sysname, "answerable": it["answerable"], "type": it.get("type"),
                             "question": it["question"], "reference": it.get("answer", ""), **out,
                             "score": score, "judge_reason": reason,
                             "cite_precision": mean(cites_ok) if cites_ok else None})
            print(f"{it['id']:<6} {sysname:<7} score={score:.1f} abstained={out['abstained']} {out['latency_s']}s")

    md = [f"# Answer evaluation\n\n{time.strftime('%Y-%m-%d %H:%M')} · model `{settings.chat_model}` · "
          f"{len(items)} items · MIN_SCORE={settings.min_score} · judge = same model (see limitations)\n",
          "| system | accuracy (answerable) | abstain acc (unanswerable) | cite precision | valid w/o repair | avg steps | avg latency (s) |",
          "|---|---|---|---|---|---|---|"]
    for s in systems:
        rows = [p for p in per_item if p["system"] == s]
        ans_rows = [p for p in rows if p["answerable"]]
        un_rows = [p for p in rows if not p["answerable"]]
        cp = [p["cite_precision"] for p in ans_rows if p["cite_precision"] is not None]
        md.append(
            f"| {s} | {mean(p['score'] for p in ans_rows):.3f} | "
            f"{mean(p['score'] for p in un_rows):.3f} | {mean(cp):.3f} | "
            f"{mean(0.0 if p['repaired'] else 1.0 for p in rows):.3f} | {mean(p['steps'] for p in rows):.2f} | "
            f"{mean(p['latency_s'] for p in rows):.2f} |"
        )
    md.append("\nPer-item outputs (use these to pick failure cases): `evaluation/results/answers_items.jsonl`")
    with (RESULTS / "answers_items.jsonl").open("w", encoding="utf-8") as f:
        for p in per_item:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    (RESULTS / "answers.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
