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
        expand_fn=None,
    ) -> list[Hit]:
        """expand_fn(query) -> hypothetical passage (HyDE). Its rankings are fused with the
        original query's rankings by RRF; the min_score floor still uses the ORIGINAL query's
        cosine so abstention behaviour does not depend on what the model imagined."""
        k = k or settings.top_k
        pool = max(pool or settings.candidate_pool, k)
        min_score = settings.min_score if min_score is None else min_score
        if mode not in ("vector", "bm25", "hybrid"):
            raise ValueError(f"unknown mode {mode!r}")
        if not query.strip() or not self.chunks:
            return []

        cos = self._cosines(query)
        queries = [(query, cos)]
        if expand_fn is not None:
            hypo = (expand_fn(query) or "").strip()
            if hypo:
                queries.append((hypo, self._cosines(hypo)))

        rank_lists: list[list[int]] = []
        for q, q_cos in queries:
            if mode in ("bm25", "hybrid"):
                bm = self.bm25.scores(q)
                rank_lists.append([int(i) for i in np.argsort(-bm, kind="stable")[:pool] if bm[i] > 0])
            if mode in ("vector", "hybrid"):
                rank_lists.append([int(i) for i in np.argsort(-q_cos, kind="stable")[:pool]])

        if len(rank_lists) == 1:  # single list: keep the raw score as rank_score
            single = bm if mode == "bm25" else cos
            ranked = [(i, float(single[i])) for i in rank_lists[0]]
        else:
            ranked = rrf(rank_lists)[:pool]

        hits = [Hit(self.chunks[i], float(cos[i]), s) for i, s in ranked]
        # min_score floor on cosine: drop weakly-related chunks so the model can abstain.
        hits = [h for h in hits if h.cosine >= min_score]
        if rerank_fn is not None and hits:
            hits = rerank_fn(query, hits)
        return hits[:k]


HYDE_PROMPT = (
    "Write ONE short passage (2-3 sentences) in the formal style of a university regulation that "
    "would answer the question. Use regulation wording such as 'Students shall ...', "
    "'not less than ... not more than ...', 'Article ...'. Invent placeholder numbers if needed "
    "(they are only used for searching, never shown to the user). Reply with the passage only."
)


def hyde_expand(query: str, chat_fn=None) -> str:
    """HyDE query expansion: questions ('how many credits can I register?') and regulations
    ('students have to enroll in not less than 9 ... not more than 22 credits') use different
    words; a hypothetical regulation sentence bridges that vocabulary gap. 1 model call."""
    from .llm import chat

    chat_fn = chat_fn or chat
    try:
        msg = [{"role": "system", "content": HYDE_PROMPT}, {"role": "user", "content": query}]
        return (chat_fn(msg).choices[0].message.content or "")[:800]
    except Exception:  # expansion is an optimisation: never let it break retrieval
        return ""


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
