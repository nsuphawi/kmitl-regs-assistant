"""Error analysis: where does the chunk containing some text rank, at every stage of retrieval?

    python -m scripts.debug_rank "How many credits can an undergraduate register in a regular semester?" --find "not more than 22 credits"

Prints the HyDE passage, the gold chunk's rank in each ranked list (BM25 / vector, for the
question and for the HyDE passage), its fused RRF rank, its cosine, and whether the
min_score floor removes it. Use the output to explain a failure case in the report.
"""
from __future__ import annotations

import argparse

import numpy as np

from app.config import settings
from app.retrieve import Retriever, hyde_expand, rrf
from app.store import Index
from evaluation.dataset import normalise


def rank_of(order: list[int], targets: set[int]) -> str:
    hits = [order.index(t) + 1 for t in targets if t in order]
    return str(min(hits)) if hits else "-"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--find", required=True, help="text that the correct chunk contains")
    ap.add_argument("--no-hyde", action="store_true")
    args = ap.parse_args()

    r = Retriever(Index.load(settings.index_dir))
    needle = normalise(args.find)
    gold = {i for i, c in enumerate(r.chunks) if needle in normalise(c.text)}
    if not gold:
        raise SystemExit(f"No chunk contains {args.find!r}")
    print("gold chunks:", [r.chunks[i].id for i in gold])

    def lists_for(q: str) -> dict[str, list[int]]:
        cos = r._cosines(q)
        bm = r.bm25.scores(q)
        return {"bm25": [int(i) for i in np.argsort(-bm, kind="stable") if bm[i] > 0],
                "vector": [int(i) for i in np.argsort(-cos, kind="stable")]}

    cos_q = r._cosines(args.question)
    full = {f"question/{k}": v for k, v in lists_for(args.question).items()}
    if not args.no_hyde:
        hypo = hyde_expand(args.question)
        print(f"\nHyDE passage:\n  {hypo}\n")
        full.update({f"hyde/{k}": v for k, v in lists_for(hypo).items()})

    print(f"{'list':<18} gold rank (of {len(r.chunks)})")
    for name, order in full.items():
        print(f"{name:<18} {rank_of(order, gold)}")

    pool = settings.candidate_pool
    for label, names in [("fused q only", [n for n in full if n.startswith("question")]), ("fused all", list(full))]:
        fused = [i for i, _ in rrf([full[n][:pool] for n in names])]
        print(f"{label:<18} {rank_of(fused, gold)}   (only the top {settings.top_k} reach the model; pool={pool})")

    for i in gold:
        c = float(cos_q[i])
        print(f"\ncosine(question, {r.chunks[i].id}) = {c:.3f}  ->  "
              f"{'REMOVED' if c < settings.min_score else 'kept'} by MIN_SCORE={settings.min_score}")


if __name__ == "__main__":
    main()
