"""Okapi BM25 in plain Python/numpy — the keyword half of hybrid search."""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
_STOP = frozenset(
    "a an and are as at be by for from has have how i in is it its of on or that the this to was what when where which who will with".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens; keeps codes like `ERR_502` or `2.00` intact, drops stop-words."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(d)) for d in docs]
        self.lens = np.array([sum(tf.values()) for tf in self.tfs], dtype=np.float32)
        self.avgdl = float(self.lens.mean()) if len(docs) else 0.0
        df: Counter[str] = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.tfs), dtype=np.float32)
        if not self.tfs:
            return out
        q_terms = set(tokenize(query))
        norm = self.k1 * (1 - self.b + self.b * self.lens / (self.avgdl or 1.0))
        for i, tf in enumerate(self.tfs):
            s = 0.0
            for w in q_terms:
                f = tf.get(w)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (f + norm[i])
            out[i] = s
        return out
