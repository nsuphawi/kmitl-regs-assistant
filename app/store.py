"""Persisted index: chunks + embedding matrix + embedder state. BM25 is rebuilt on load (cheap)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .embeddings import embedder_from_state, make_embedder
from .ingest import Chunk, chunk_pages, load_pages


@dataclass
class Index:
    chunks: list[Chunk]
    embeddings: np.ndarray
    embedder: object
    meta: dict

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        with (path / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        np.save(path / "embeddings.npy", self.embeddings)
        (path / "meta.json").write_text(
            json.dumps({**self.meta, "embedder": self.embedder.state()}, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "Index":
        if not (path / "meta.json").exists():
            raise FileNotFoundError(f"No index at {path}. Run: python -m scripts.build_index")
        chunks = [Chunk(**json.loads(line)) for line in (path / "chunks.jsonl").open(encoding="utf-8")]
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        embedder = embedder_from_state(meta["embedder"])
        return cls(chunks, np.load(path / "embeddings.npy"), embedder, meta)


def build_index(raw_dir: Path, chunk_words: int, overlap: int, provider: str | None = None, model: str | None = None) -> Index:
    pages = load_pages(raw_dir)
    if not pages:
        raise RuntimeError(f"No documents found in {raw_dir}. Run: python -m scripts.fetch_corpus")
    chunks = chunk_pages(pages, chunk_words, overlap)
    texts = [c.text for c in chunks]
    embedder = make_embedder(provider, model, corpus=texts)
    emb = embedder.embed(texts)
    meta = {"chunk_words": chunk_words, "chunk_overlap": overlap, "n_chunks": len(chunks)}
    return Index(chunks, emb, embedder, meta)
