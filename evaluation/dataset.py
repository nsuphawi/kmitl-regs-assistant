"""Eval-set loading and chunk-size-independent gold labels.

Gold is stored as EVIDENCE QUOTES (verbatim text from the PDF), not chunk ids. A chunk is
relevant if it contains the quote. That way the same labels work for every chunk size,
which is what makes the chunk-size experiment fair.

Line format of data/eval/qa.jsonl:
  {"id": "q01", "question": "...", "answer": "reference answer",
   "evidence": ["verbatim quote from the document"], "doc": "KMITL_Undergraduate_Regulations_2016",
   "answerable": true, "type": "factual|numeric|keyword|paraphrase|unanswerable", "verified": true}
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from app.ingest import Chunk

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalise(text: str) -> str:
    return _NON_ALNUM.sub(" ", text.lower()).strip()


def load_eval(path: Path, include_unverified: bool = False) -> list[dict]:
    items = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        item = json.loads(line)
        for key in ("id", "question", "answerable"):
            if key not in item:
                raise ValueError(f"{path}:{n} missing '{key}'")
        if item.get("verified") or include_unverified:
            items.append(item)
    return items


def _contains(chunk_norm: str, quote_norm: str, edge_words: int = 10) -> bool:
    if quote_norm in chunk_norm:
        return True
    # Quote split across a chunk boundary: accept a chunk holding its first or last N words.
    words = quote_norm.split()
    if len(words) <= edge_words:
        return False
    return " ".join(words[:edge_words]) in chunk_norm or " ".join(words[-edge_words:]) in chunk_norm


def relevant_ids(item: dict, chunks: list[Chunk], _cache: dict | None = None) -> set[str]:
    if not item.get("answerable", True):
        return set()
    quotes = [normalise(q) for q in item.get("evidence", []) if q.strip()]
    doc = item.get("doc")
    out = set()
    for c in chunks:
        if doc and c.doc_id != doc:
            continue
        cn = normalise(c.text)
        if any(_contains(cn, q) for q in quotes):
            out.add(c.id)
    return out
