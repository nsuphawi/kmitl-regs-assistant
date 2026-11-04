"""Draft candidate eval questions from random chunks — YOU must review every one by hand.

    python -m scripts.draft_eval --n 40

For each sampled chunk the model proposes {question, answer, evidence}. The evidence quote is
checked to really occur in the chunk; drafts that fail are discarded. Output goes to
data/eval/qa_draft.jsonl with "verified": false.

Review checklist (per line): Is the question something a real student would ask? Is the answer
correct per the PDF? Is the question PARAPHRASED (not copied from the evidence — copying inflates
BM25)? Then set "verified": true and append it to data/eval/qa.jsonl.
"""
from __future__ import annotations

import argparse
import json
import random

from pydantic import BaseModel, Field, ValidationError

from app.config import ROOT, settings
from app.llm import chat, extract_json
from app.store import Index
from evaluation.dataset import normalise

PROMPT = (
    "You write evaluation questions for a KMITL student-regulations assistant. From the passage, "
    "write ONE question a student would realistically ask that this passage answers. Paraphrase: do "
    "not reuse the passage's distinctive phrases in the question. Then give a short reference answer "
    "and an EXACT evidence quote (10-35 words) copied character-for-character from the passage.\n"
    'Reply with JSON only: {"question": "...", "answer": "...", "evidence": "...", '
    '"type": "factual|numeric|keyword|paraphrase"}\n'
    'If the passage has no answerable fact (e.g. a table of contents), reply {"skip": true}.'
)


class Draft(BaseModel):
    question: str = Field(..., min_length=10, max_length=300)
    answer: str = Field(..., min_length=1, max_length=600)
    evidence: str = Field(..., min_length=20, max_length=400)
    type: str = "factual"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    idx = Index.load(settings.index_dir)
    rng = random.Random(args.seed)
    candidates = [c for c in idx.chunks if len(c.text.split()) > 80]
    rng.shuffle(candidates)
    out_path = ROOT / "data" / "eval" / "qa_draft.jsonl"
    kept = 0
    with out_path.open("w", encoding="utf-8") as f:
        for c in candidates:
            if kept >= args.n:
                break
            msg = [{"role": "system", "content": PROMPT}, {"role": "user", "content": c.text}]
            try:
                obj = extract_json(chat(msg).choices[0].message.content)
                if obj.get("skip"):
                    continue
                d = Draft.model_validate(obj)
            except (ValueError, ValidationError, AttributeError):
                continue
            if normalise(d.evidence) not in normalise(c.text):
                print(f"  drop {c.id}: evidence not verbatim")
                continue
            kept += 1
            f.write(json.dumps({"id": f"d{kept:02d}", "question": d.question, "answer": d.answer,
                                "evidence": [d.evidence], "doc": c.doc_id, "answerable": True,
                                "type": d.type, "source_chunk": c.id, "verified": False},
                               ensure_ascii=False) + "\n")
            print(f"  kept {c.id}: {d.question}")
    print(f"\nWrote {kept} drafts to {out_path}. Review each one, set \"verified\": true, append to qa.jsonl.")


if __name__ == "__main__":
    main()
