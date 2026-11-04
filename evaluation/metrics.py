"""Retrieval metrics (Lecture 8 / HW4). Pure functions, stdlib only."""
from __future__ import annotations


def recall_at_k(retrieved: list[str], relevant, k: int) -> float:
    relevant = set(relevant)
    if not relevant:
        return 0.0
    return len(relevant & set(retrieved[:k])) / len(relevant)


def hit_at_k(retrieved: list[str], relevant, k: int) -> float:
    """1.0 if ANY relevant chunk is in the top-k. With overlapping chunks a fact often lives in
    two neighbouring chunks, and the model only needs one, so this is the headline metric."""
    relevant = set(relevant)
    return 1.0 if relevant & set(retrieved[:k]) else 0.0


def precision_at_k(retrieved: list[str], relevant, k: int) -> float:
    if k <= 0:
        return 0.0
    relevant = set(relevant)
    return len(relevant & set(retrieved[:k])) / k


def reciprocal_rank(retrieved: list[str], relevant) -> float:
    relevant = set(relevant)
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0
