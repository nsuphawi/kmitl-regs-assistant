"""Entry point for Hugging Face Spaces (Gradio SDK, free CPU tier) or any plain Python host.

Spaces runs `python app.py` and expects a web server on port 7860. This file builds the index
if it is missing, then serves the project's own FastAPI app (JSON API, SSE stream, web UI).
`import app.api` below resolves to the `app/` package, which takes precedence over this file.

Settings come from the environment (Space secrets / variables):
    OPENAI_API_KEY (secret), OPENAI_BASE_URL, CHAT_MODEL
"""
from __future__ import annotations

import json
import os

import uvicorn

from app.config import settings
from app.store import build_index


def _index_matches() -> bool:
    """An index built with another embedder or chunking (e.g. local MiniLM) can't be reused."""
    meta_path = settings.index_dir / "meta.json"
    if not meta_path.exists():
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    emb = meta.get("embedder", {})
    same_embedder = emb.get("provider") == settings.embed_provider and (
        settings.embed_provider == "tfidf" or emb.get("model") == settings.embed_model)
    same_chunks = (meta.get("chunk_strategy"), meta.get("chunk_words")) == (settings.chunk_strategy, settings.chunk_words)
    return same_embedder and same_chunks


def ensure_index() -> None:
    if _index_matches():
        print(f"Using existing index at {settings.index_dir}")
        return
    print("Building index (first start only)...")
    index = build_index(settings.raw_dir, settings.chunk_words, settings.chunk_overlap,
                        settings.embed_provider, strategy=settings.chunk_strategy)
    index.save(settings.index_dir)
    print(f"Indexed {len(index.chunks)} chunks -> {settings.index_dir}")


if __name__ == "__main__":
    import threading

    from app import api

    if _index_matches():
        print(f"Using existing index at {settings.index_dir}")
    else:
        # Open the port at once (hosts kill services that don't), build in the background.
        def build() -> None:
            ensure_index()
            from app.retrieve import Retriever
            from app.store import Index

            api.STATE["retriever"] = Retriever(Index.load(settings.index_dir))
            print("Index ready; the app is serving questions.", flush=True)

        threading.Thread(target=build, daemon=True).start()

    uvicorn.run(api.app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
