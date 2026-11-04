"""Retrieval evaluation: baselines vs hybrid, chunk-size sweep, and min_score sweep.

    python -m evaluation.run_retrieval_eval
    python -m evaluation.run_retrieval_eval --chunk-sizes 150,300,600 --rerank

No LLM calls unless --rerank. Writes evaluation/results/retrieval.md and retrieval.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from app.config import ROOT, settings
from app.retrieve import Retriever, hyde_expand, llm_rerank
from app.store import Index, build_index

from .dataset import load_eval, relevant_ids
from .metrics import hit_at_k, mean, recall_at_k, reciprocal_rank

RESULTS = ROOT / "evaluation" / "results"
KS = (1, 3, 5)


def get_index(chunk_words: int, overlap: int, provider: str, rebuild: bool, strategy: str = "window") -> Index:
    """Cache one index per (chunk size, provider) so sweeps are reproducible and fast."""
    path = ROOT / "data" / "index_cache" / f"{strategy}_cs{chunk_words}_ov{overlap}_{provider}"
    if path.exists() and not rebuild:
        return Index.load(path)
    idx = build_index(settings.raw_dir, chunk_words, overlap, provider, strategy=strategy)
    idx.save(path)
    return idx


def score_mode(r: Retriever, items: list[dict], gold: dict[str, set], mode: str, rerank: bool = False, hyde: bool = False) -> dict:
    rows = []
    for it in items:
        rel = gold[it["id"]]
        hits = r.search(it["question"], k=max(KS), mode=mode, min_score=-1.0,
                        rerank_fn=llm_rerank if rerank else None, expand_fn=hyde_expand if hyde else None)
        ids = [h.chunk.id for h in hits]
        rows.append({"id": it["id"], "retrieved": ids, "relevant": sorted(rel),
                     **{f"hit@{k}": hit_at_k(ids, rel, k) for k in KS},
                     **{f"recall@{k}": recall_at_k(ids, rel, k) for k in KS},
                     "rr": reciprocal_rank(ids, rel)})
    summary = {f"hit@{k}": mean(r_[f"hit@{k}"] for r_ in rows) for k in KS}
    summary.update({f"recall@{k}": mean(r_[f"recall@{k}"] for r_ in rows) for k in KS})
    summary["mrr"] = mean(r_["rr"] for r_ in rows)
    return {"summary": summary, "rows": rows}


def min_score_sweep(r: Retriever, answerable: list[dict], unanswerable: list[dict], gold, thresholds) -> list[dict]:
    out = []
    for t in thresholds:
        kept = mean(hit_at_k([h.chunk.id for h in r.search(it["question"], k=5, min_score=t)], gold[it["id"]], 5)
                    for it in answerable)
        abstain = mean(1.0 if not r.search(it["question"], k=5, min_score=t) else 0.0 for it in unanswerable)
        out.append({"min_score": t, "answerable_hit@5": kept, "unanswerable_abstain": abstain,
                    "balanced": (kept + abstain) / 2})
    return out


def fmt(x: float) -> str:
    return f"{x:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", type=Path, default=ROOT / "data" / "eval" / "qa.jsonl")
    ap.add_argument("--chunk-sizes", default="100,150,300")
    ap.add_argument("--strategies", default="window,clause", help="chunking strategies to compare")
    ap.add_argument("--overlap-ratio", type=float, default=1 / 6)
    ap.add_argument("--provider", default=settings.embed_provider)
    ap.add_argument("--thresholds", default="0.0,0.2,0.3,0.35,0.4,0.45,0.5,0.55,0.6")
    ap.add_argument("--rerank", action="store_true", help="also evaluate hybrid + LLM rerank (slow: 1 call/chunk)")
    ap.add_argument("--hyde", action="store_true", help="also evaluate hybrid + HyDE query expansion (1 call/question)")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--include-unverified", action="store_true")
    args = ap.parse_args()

    items = load_eval(args.qa, args.include_unverified)
    answerable = [it for it in items if it["answerable"]]
    unanswerable = [it for it in items if not it["answerable"]]
    print(f"Eval set: {len(items)} items ({len(answerable)} answerable, {len(unanswerable)} unanswerable)")
    if len(items) < 30:
        print("WARNING: fewer than 30 verified items — the project spec asks for ~30.")

    sizes = [int(s) for s in args.chunk_sizes.split(",")]
    modes = ["bm25", "vector", "hybrid"]
    report = {"provider": args.provider, "n_answerable": len(answerable), "n_unanswerable": len(unanswerable),
              "runs": {}, "min_score_sweep": [], "label_problems": []}
    md = [f"# Retrieval evaluation\n\n{time.strftime('%Y-%m-%d %H:%M')} · embeddings: `{args.provider}` · "
          f"{len(answerable)} answerable / {len(unanswerable)} unanswerable questions\n",
          "| chunking | words | mode | hit@1 | hit@3 | hit@5 | recall@5 | MRR |", "|---|---|---|---|---|---|---|---|"]

    default_r, default_gold = None, None
    configs = [(st, sz) for st in args.strategies.split(",") for sz in sizes]
    for strategy, size in configs:
        idx = get_index(size, max(0, round(size * args.overlap_ratio)), args.provider, args.rebuild, strategy)
        r = Retriever(idx)
        gold = {it["id"]: relevant_ids(it, idx.chunks) for it in items}
        missing = [it["id"] for it in answerable if not gold[it["id"]]]
        if missing:
            print(f"  [{strategy}/{size}] evidence not found in any chunk for: {missing} (fix the quote or doc name)")
            report["label_problems"].append({"strategy": strategy, "chunk_words": size, "ids": missing})
        extra = []
        is_default = strategy == settings.chunk_strategy and size == settings.chunk_words
        if is_default:
            extra += ["hybrid+hyde"] if args.hyde else []
            extra += ["hybrid+rerank"] if args.rerank else []
        for mode in modes + extra:
            res = score_mode(r, answerable, gold, mode.split("+")[0], rerank=mode.endswith("rerank"), hyde=mode.endswith("hyde"))
            report["runs"][f"{strategy}/{size}/{mode}"] = res
            s = res["summary"]
            md.append(f"| {strategy} | {size} | {mode} | {fmt(s['hit@1'])} | {fmt(s['hit@3'])} | {fmt(s['hit@5'])} | {fmt(s['recall@5'])} | {fmt(s['mrr'])} |")
            print(f"  {strategy:<7} {size:<4} {mode:<14} hit@1={fmt(s['hit@1'])} hit@5={fmt(s['hit@5'])} MRR={fmt(s['mrr'])}")
        if is_default or default_r is None:
            default_r, default_gold, default_size = r, gold, f"{strategy} {size}"

    if unanswerable:
        thresholds = [float(t) for t in args.thresholds.split(",")]
        sweep = min_score_sweep(default_r, answerable, unanswerable, default_gold, thresholds)
        report["min_score_sweep"] = sweep
        best = max(sweep, key=lambda row: (row["balanced"], row["answerable_hit@5"]))
        md += [f"\n## min_score sweep (hybrid, {default_size}-word chunks)\n",
               "| min_score | answerable hit@5 | unanswerable correctly abstained | balanced |", "|---|---|---|---|"]
        md += [f"| {row['min_score']} | {fmt(row['answerable_hit@5'])} | {fmt(row['unanswerable_abstain'])} | {fmt(row['balanced'])} |" for row in sweep]
        md.append(f"\nBest balanced threshold: **{best['min_score']}** — set `MIN_SCORE={best['min_score']}` in `.env`.")
        print(f"  min_score sweep: best balanced = {best['min_score']} ({fmt(best['balanced'])})")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "retrieval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (RESULTS / "retrieval.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nWrote {RESULTS / 'retrieval.md'}")


if __name__ == "__main__":
    main()
