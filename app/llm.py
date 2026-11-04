"""Thin wrapper around the class proxy. One place to change model / client."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any, Callable

from .config import settings

ChatFn = Callable[..., Any]


@lru_cache(maxsize=1)
def get_client():
    from openai import OpenAI

    return OpenAI()  # reads OPENAI_API_KEY / OPENAI_BASE_URL from the environment


def chat(messages: list[dict], **kwargs) -> Any:
    """Deterministic chat call (temperature=0) — everything we evaluate must be repeatable."""
    kwargs.setdefault("temperature", 0)
    return get_client().chat.completions.create(
        model=settings.chat_model, messages=messages, **kwargs
    )


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def extract_json(text: str) -> Any:
    """Parse a JSON object from model text: strip ``` fences, else grab the outermost {...}.

    Raises ValueError when nothing parseable is found.
    """
    if not isinstance(text, str):
        raise ValueError("model returned no text")
    s = _FENCE.sub("", text.strip()).strip()
    # strict=False: small models often put raw newlines (bullet lists) inside JSON strings
    try:
        return json.loads(s, strict=False)
    except json.JSONDecodeError:
        pass
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(s[start : end + 1], strict=False)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON: {exc}") from exc
    raise ValueError("no JSON object found")
