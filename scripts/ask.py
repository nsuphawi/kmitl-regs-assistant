"""Command-line client, handy for the live demo.

    python -m scripts.ask "How many credits can I register per regular semester?"
    python -m scripts.ask "Hi!" --mode agent
    python -m scripts.ask "..." --mode rag --show-context
"""
from __future__ import annotations

import argparse
import json

from app.agent import run_agent_events
from app.answer import answer_closed_book, answer_rag
from app.config import settings
from app.retrieve import Retriever
from app.store import Index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--mode", choices=["agent", "rag", "closed"], default="agent")
    ap.add_argument("--k", type=int, default=settings.top_k)
    ap.add_argument("--show-context", action="store_true")
    args = ap.parse_args()

    if args.mode == "closed":
        print(json.dumps(answer_closed_book(args.question).answer.model_dump(), indent=2, ensure_ascii=False))
        return
    r = Retriever(Index.load(settings.index_dir))
    if args.mode == "rag":
        res = answer_rag(args.question, r, k=args.k)
        if args.show_context:
            for h in res.hits:
                print(f"--- [{h.chunk.id}] cos={h.cosine:.3f} {h.chunk.cite}\n{h.chunk.text[:500]}\n")
        print(json.dumps(res.answer.model_dump(), indent=2, ensure_ascii=False))
        return
    for ev in run_agent_events(args.question, r):
        if ev["type"] == "final":
            print(json.dumps({k: ev[k] for k in ("answer", "sources", "steps", "stopped")}, indent=2, ensure_ascii=False))
        elif ev["type"] != "start":
            print(f"[{ev['type']}] " + json.dumps({k: v for k, v in ev.items() if k != "type"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
