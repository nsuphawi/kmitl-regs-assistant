"""Chunk + embed the corpus and save the index used by the app.

    python -m scripts.build_index                         # uses CHUNK_WORDS / EMBED_PROVIDER from env
    python -m scripts.build_index --chunk-words 200 --out data/index_cs200
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from app.config import settings
from app.store import build_index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-words", type=int, default=settings.chunk_words)
    ap.add_argument("--overlap", type=int, default=settings.chunk_overlap)
    ap.add_argument("--strategy", choices=["clause", "window"], default=settings.chunk_strategy)
    ap.add_argument("--provider", default=settings.embed_provider)
    ap.add_argument("--out", type=Path, default=settings.index_dir)
    args = ap.parse_args()

    t0 = time.time()
    index = build_index(settings.raw_dir, args.chunk_words, args.overlap, args.provider, strategy=args.strategy)
    index.save(args.out)
    docs = sorted({c.doc_id for c in index.chunks})
    print(f"[{args.strategy}, {args.chunk_words} words] Indexed {len(index.chunks)} chunks from {len(docs)} documents in {time.time() - t0:.1f}s -> {args.out}")
    for d in docs:
        print(f"  {d}: {sum(c.doc_id == d for c in index.chunks)} chunks")


if __name__ == "__main__":
    main()
