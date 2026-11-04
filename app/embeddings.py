"""Embedding back-ends. All return L2-normalised float32 rows, so dot product == cosine."""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from .config import settings

_TOKEN = re.compile(r"[a-z0-9]+")


def _normalise(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    if m.ndim == 1:
        m = m[None, :]
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class LocalEmbedder:
    """sentence-transformers model run locally (default)."""

    def __init__(self, model: str):
        from sentence_transformers import SentenceTransformer

        self.name = model
        self._model = SentenceTransformer(model)

    def embed(self, texts: list[str]) -> np.ndarray:
        return _normalise(self._model.encode(texts, batch_size=32, show_progress_bar=False))

    def state(self) -> dict:
        return {"provider": "local", "model": self.name}


class OpenAIEmbedder:
    """The proxy's /embeddings endpoint (only if your proxy serves an embedding model)."""

    def __init__(self, model: str):
        from .llm import get_client

        self.name = model
        self._client = get_client()

    def embed(self, texts: list[str]) -> np.ndarray:
        rows: list[list[float]] = []
        for i in range(0, len(texts), 64):
            resp = self._client.embeddings.create(model=self.name, input=texts[i : i + 64])
            rows.extend(d.embedding for d in resp.data)
        return _normalise(np.array(rows))

    def state(self) -> dict:
        return {"provider": "openai", "model": self.name}


class TfidfEmbedder:
    """Dependency-free sparse-ish TF-IDF vectors. Weak, but deterministic and offline."""

    def __init__(self, vocab: dict[str, int] | None = None, idf: list[float] | None = None):
        self.vocab = vocab or {}
        self.idf = np.array(idf or [], dtype=np.float32)
        self.name = "tfidf"

    def fit(self, texts: list[str]) -> "TfidfEmbedder":
        df: Counter[str] = Counter()
        for t in texts:
            df.update(set(_TOKEN.findall(t.lower())))
        self.vocab = {w: i for i, w in enumerate(sorted(df))}
        n = len(texts)
        self.idf = np.array([math.log((1 + n) / (1 + df[w])) + 1 for w in sorted(df)], dtype=np.float32)
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        m = np.zeros((len(texts), max(len(self.vocab), 1)), dtype=np.float32)
        for r, t in enumerate(texts):
            for w, c in Counter(_TOKEN.findall(t.lower())).items():
                j = self.vocab.get(w)
                if j is not None:
                    m[r, j] = (1 + math.log(c)) * self.idf[j]
        return _normalise(m)

    def state(self) -> dict:
        return {"provider": "tfidf", "model": "tfidf", "vocab": self.vocab, "idf": self.idf.tolist()}


def make_embedder(provider: str | None = None, model: str | None = None, corpus: list[str] | None = None):
    provider = provider or settings.embed_provider
    model = model or settings.embed_model
    if provider == "local":
        return LocalEmbedder(model)
    if provider == "openai":
        return OpenAIEmbedder(model)
    if provider == "tfidf":
        if corpus is None:
            raise ValueError("tfidf embedder needs the corpus to fit on")
        return TfidfEmbedder().fit(corpus)
    raise ValueError(f"unknown EMBED_PROVIDER: {provider!r}")


def embedder_from_state(state: dict):
    if state["provider"] == "tfidf":
        return TfidfEmbedder(state["vocab"], state["idf"])
    return make_embedder(state["provider"], state["model"])
