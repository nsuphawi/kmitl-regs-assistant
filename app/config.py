"""Central configuration. Everything comes from the environment — no secrets in code.

The OpenAI client itself reads OPENAI_API_KEY / OPENAI_BASE_URL (the class proxy),
so they are deliberately NOT read or stored here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

try:  # optional: load a local .env (never committed) so `python -m ...` just works
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
except ImportError:
    pass


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    chat_model: str = _env("CHAT_MODEL", "gemma-4-E4B-it")
    # local  = sentence-transformers on your machine (default, works without proxy embeddings)
    # openai = the proxy's /embeddings endpoint (set EMBED_MODEL to a model it serves)
    # tfidf  = pure-numpy fallback, no downloads (used by the unit tests)
    embed_provider: str = _env("EMBED_PROVIDER", "local")
    embed_model: str = _env("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # clause = split at Article/clause numbers (structure-aware); window = naive fixed window
    chunk_strategy: str = _env("CHUNK_STRATEGY", "clause")
    chunk_words: int = int(_env("CHUNK_WORDS", "150"))
    chunk_overlap: int = int(_env("CHUNK_OVERLAP", "25"))
    top_k: int = int(_env("TOP_K", "5"))
    candidate_pool: int = int(_env("CANDIDATE_POOL", "20"))
    # Cosine floor below which a chunk is dropped. Pick it with
    # `python -m evaluation.run_retrieval_eval` (min_score sweep), not by guessing.
    min_score: float = float(_env("MIN_SCORE", "0.40"))
    max_steps: int = int(_env("MAX_STEPS", "5"))
    # HyDE query expansion (1 extra model call per search). Keep it only if
    # `python -m evaluation.run_retrieval_eval --hyde` shows it helps.
    hyde: bool = _env("HYDE", "1").lower() in {"1", "true", "yes"}

    raw_dir: Path = Path(_env("RAW_DIR", str(ROOT / "data" / "raw")))
    index_dir: Path = Path(_env("INDEX_DIR", str(ROOT / "data" / "index")))


settings = Settings()
