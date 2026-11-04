"""Shared fixtures: a tiny fictional corpus indexed with the offline TF-IDF embedder,
and a scripted fake LLM so tests never touch the network."""
from __future__ import annotations

import os

os.environ["HYDE"] = "0"  # before importing app: scripted fake LLM must not be consumed by HyDE

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.retrieve import Retriever
from app.store import build_index

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def index(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    shutil.copy(FIXTURES / "example_handbook.txt", raw / "example_handbook.txt")
    return build_index(raw, chunk_words=40, overlap=8, provider="tfidf")


@pytest.fixture(scope="session")
def retriever(index):
    return Retriever(index)


def _tool_call(name: str, args: dict, call_id: str = "c1"):
    fn = SimpleNamespace(name=name, arguments=json.dumps(args))
    return SimpleNamespace(id=call_id, type="function", function=fn)


class FakeLLM:
    """Returns scripted replies in order. Each reply is a str (content) or a list of tool calls."""

    def __init__(self, replies, reject_tools: bool = False):
        self.replies = list(replies)
        self.calls: list[dict] = []
        self.reject_tools = reject_tools

    def __call__(self, messages, **kwargs):
        if self.reject_tools and "tools" in kwargs:
            raise RuntimeError("tools not supported")
        self.calls.append({"messages": [dict(m) for m in messages], **kwargs})
        reply = self.replies.pop(0) if self.replies else '{"answer": "fallback", "citations": [], "abstained": true}'
        if isinstance(reply, list):
            msg = SimpleNamespace(content=None, tool_calls=[_tool_call(*tc) for tc in reply])
        else:
            msg = SimpleNamespace(content=reply, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


@pytest.fixture
def fake_llm():
    return FakeLLM
