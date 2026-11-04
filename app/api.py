"""FastAPI backend: JSON endpoint + SSE stream of the agent's steps + a tiny web UI.

Run:  uvicorn app.api:app --reload      then open http://127.0.0.1:8000
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from .agent import run_agent, run_agent_events
from .answer import answer_rag
from .config import settings
from .retrieve import Retriever
from .schemas import AskRequest, AskResponse
from .store import Index

STATE: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    STATE["retriever"] = Retriever(Index.load(settings.index_dir))
    yield
    STATE.clear()


app = FastAPI(title="KMITL Regulations Assistant", lifespan=lifespan)


def _retriever() -> Retriever:
    r = STATE.get("retriever")
    if r is None:
        raise HTTPException(503, "index not loaded")
    return r


@app.get("/health")
def health() -> dict:
    r = STATE.get("retriever")
    return {"ok": r is not None, "chunks": len(r.chunks) if r else 0, "model": settings.chat_model,
            "embeddings": r.index.meta.get("embedder", {}).get("model") if r else None}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    r = _retriever()
    if req.mode == "rag":
        res = await run_in_threadpool(answer_rag, req.question, r, req.k)
        cited = {c.chunk_id for c in res.answer.citations}
        sources = [h.to_dict(max_chars=400) for h in res.hits if h.chunk.id in cited]
        return AskResponse(answer=res.answer, sources=sources)
    res = await run_in_threadpool(run_agent, req.question, r)
    final = res.trace[-1]
    return AskResponse(answer=res.answer, sources=final["sources"], trace=res.trace[:-1])


@app.get("/ask/stream")
async def ask_stream(question: str = Query(..., min_length=3, max_length=500)) -> StreamingResponse:
    r = _retriever()

    def events():
        try:
            for ev in run_agent_events(question, r):
                yield f"event: {ev['type']}\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except Exception as exc:  # surface failures to the client instead of a dead stream
            yield f"event: error\ndata: {json.dumps({'type': 'error', 'message': str(exc)[:300]})}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(iterate_in_threadpool(events()), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/")
def index() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")
