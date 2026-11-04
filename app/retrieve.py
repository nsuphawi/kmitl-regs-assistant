"""Retrieval: BM25, vector, hybrid (RRF), optional LLM rerank, and a min_score floor."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .bm25 import BM25
from .config import settings
from .ingest import Chunk
from .store import Index

Mode = Literal["bm25", "vector", "hybrid"]


@dataclass
class Hit:
    chunk: Chunk
    cosine: float  # always the vector similarity, so min_score means the same thing in every mode
    rank_score: float  # the score the ranking was made with (bm25 / cosine / rrf / rerank)

    def to_dict(self, max_chars: int | None = None) -> dict:
        text = self.chunk.text if max_chars is None else self.chunk.text[:max_chars]
        return {
            "chunk_id": self.chunk.id,
            "source": self.chunk.cite,
            "cosine": round(self.cosine, 4),
            "score": round(self.rank_score, 4),
            "text": text,
        }


def rrf(rank_lists: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion (HW4 Q5): fuse by rank, never raw score. Stable on ties."""
    scores: dict[int, float] = {}
    for ranking in rank_lists:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])


class Retriever:
    def __init__(self, index: Index):
        self.index = index
        self.chunks = index.chunks
        self.bm25 = BM25([c.text for c in self.chunks])

    def _cosines(self, query: str) -> np.ndarray:
        q = self.index.embedder.embed([query])[0]
        return self.index.embeddings @ q

    def search(
        self,
        query: str,
        k: int | None = None,
        mode: Mode = "hybrid",
        min_score: float | None = None,
        pool: int | None = None,
        rerank_fn=None,
    ) -> list[Hit]:
        k = k or settings.top_k
        pool = max(pool or settings.candidate_pool, k)
        min_score = settings.min_score if min_score is None else min_score
        if not query.strip() or not self.chunks:
            return []

        cos = self._cosines(query)
        if mode == "vector":
            order = np.argsort(-cos, kind="stable")[:pool]
            ranked = [(int(i), float(cos[i])) for i in order]
        elif mode == "bm25":
            bm = self.bm25.scores(query)
            order = [int(i) for i in np.argsort(-bm, kind="stable")[:pool] if bm[i] > 0]
            ranked = [(i, float(bm[i])) for i in order]
        elif mode == "hybrid":
            bm = self.bm25.scores(query)
            bm_rank = [int(i) for i in np.argsort(-bm, kind="stable")[:pool] if bm[i] > 0]
            vec_rank = [int(i) for i in np.argsort(-cos, kind="stable")[:pool]]
            ranked = rrf([bm_rank, vec_rank])[:pool]
        else:
            raise ValueError(f"unknown mode {mode!r}")

        hits = [Hit(self.chunks[i], float(cos[i]), s) for i, s in ranked]
        # min_score floor on cosine: drop weakly-related chunks so the model can abstain.
        hits = [h for h in hits if h.cosine >= min_score]
        if rerank_fn is not None and hits:
            hits = rerank_fn(query, hits)
        return hits[:k]


def llm_rerank(query: str, hits: list[Hit], chat_fn=None) -> list[Hit]:
    """'Ask the model to rate each chunk 1-10' reranker (Lecture 8). One call per chunk."""
    from .llm import chat, extract_json

    chat_fn = chat_fn or chat
    rescored: list[Hit] = []
    for h in hits:
        msg = [
            {"role": "system", "content": 'Rate how well the passage answers the question. Reply with JSON only: {"score": <integer 1-10>}'},
            {"role": "user", "content": f"Question: {query}\n\nPassage:\n{h.chunk.text[:1500]}"},
        ]
        try:
            raw = chat_fn(msg).choices[0].message.content
            score = float(extract_json(raw)["score"])
            score = min(max(score, 1.0), 10.0)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            score = 0.0  # unparseable rating -> push to the bottom, never crash
        rescored.append(Hit(h.chunk, h.cosine, score))
    return sorted(rescored, key=lambda h: -h.rank_score)  # stable: ties keep fused order
