"""Embedding back-ends. All return L2-normalised float32 rows, so dot product == cosine."""
from __future__ import annotations

import math
import os
import re
import time
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

    # Free tiers count every text as one request (Gemini: 100 per minute), so pace ourselves.
    rpm = int(os.getenv("EMBED_RPM", "90"))
    _sent: list[float] = []  # timestamps of texts sent, shared by all instances

    def _wait_for(self, n: int) -> None:
        while True:
            now = time.time()
            self._sent[:] = [t for t in self._sent if now - t < 60]
            if len(self._sent) + n <= self.rpm:
                self._sent.extend([now] * n)
                return
            wait = 60 - (now - self._sent[0]) + 0.5
            print(f"  embedding rate limit: waiting {wait:.0f}s", flush=True)
            time.sleep(wait)

    def embed(self, texts: list[str]) -> np.ndarray:
        rows: list[list[float]] = []
        batch = max(1, min(32, self.rpm))
        for i in range(0, len(texts), batch):
            part = texts[i : i + batch]
            for attempt in range(6):
                self._wait_for(len(part))
                try:
                    resp = self._client.embeddings.create(model=self.name, input=part)
                    break
                except Exception as exc:
                    msg = str(exc)
                    if attempt == 5 or ("429" not in msg and "rate" not in msg.lower()):
                        raise
                    m = re.search(r"retry in ([0-9.]+)s", msg)
                    time.sleep(float(m.group(1)) + 1 if m else 15)
            rows.extend(d.embedding for d in resp.data)
            if len(texts) > batch:
                print(f"  embedded {min(i + batch, len(texts))}/{len(texts)}", flush=True)
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
