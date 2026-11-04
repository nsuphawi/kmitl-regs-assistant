"""Pydantic schemas for everything that crosses a boundary (model output, API I/O)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Citation(BaseModel):
    chunk_id: str = Field(..., max_length=120)
    quote: str = Field("", max_length=400, description="Short supporting quote copied from the chunk")

    @field_validator("chunk_id", mode="before")
    @classmethod
    def _clean_id(cls, v):
        # models often wrap ids as "[doc:0013]" or add spaces
        return v.strip().strip("[]()").strip() if isinstance(v, str) else v

    @field_validator("quote", mode="before")
    @classmethod
    def _trim_quote(cls, v):
        # a too-long quote is not a reason to throw away a correct answer: truncate it
        return v[:400] if isinstance(v, str) else ("" if v is None else v)


class RAGAnswer(BaseModel):
    """The structured answer the model must return (validated, repaired once on failure)."""

    answer: str = Field(..., max_length=4000)
    citations: list[Citation] = Field(default_factory=list, max_length=8)
    abstained: bool = False

    @field_validator("citations", mode="before")
    @classmethod
    def _coerce_citations(cls, v):
        if v is None:
            return []
        if isinstance(v, list):  # accept ["doc:0013", ...] as well as [{"chunk_id": ...}, ...]
            v = [{"chunk_id": c} if isinstance(c, str) else c for c in v]
            return v[:8]
        return v

    @field_validator("answer", mode="before")
    @classmethod
    def _trim_answer(cls, v):
        return v[:4000] if isinstance(v, str) else v

    @field_validator("answer")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("answer must not be empty")
        return v.strip()


ANSWER_JSON_SPEC = (
    'Reply with ONE JSON object and nothing else:\n'
    '{"answer": "<your answer>", '
    '"citations": [{"chunk_id": "<id from the context>", "quote": "<short exact quote>"}], '
    '"abstained": false}\n'
    'If the context does not contain the answer, reply '
    '{"answer": "I could not find this in the KMITL documents.", "citations": [], "abstained": true}.'
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    mode: Literal["rag", "agent"] = "agent"
    k: int = Field(5, ge=1, le=10)


class AskResponse(BaseModel):
    answer: RAGAnswer
    sources: list[dict]
    trace: list[dict] = Field(default_factory=list)
