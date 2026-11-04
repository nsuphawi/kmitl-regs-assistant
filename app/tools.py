"""Allow-listed agent tools: schemas, argument validation, and safe execution.

Rule: the model only *requests* a tool; this module decides whether the request is legal.
"""
from __future__ import annotations

import ast
import operator
from typing import Any, Callable

from .retrieve import Retriever


class ToolError(Exception):
    """A rejected or failed tool call. The message is fed back to the model."""


# ---------- calculator: AST walk over a tiny numeric grammar (no eval/exec) ----------
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS: dict[str, Callable[..., float]] = {"round": round, "min": min, "max": max, "abs": abs, "sum": lambda *a: sum(a)}


def safe_calc(expression: str) -> float:
    def ev(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and (abs(right) > 10 or abs(left) > 1e6):
                raise ToolError("exponent too large")
            return _BIN[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](ev(node.operand))
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS
                and not node.keywords and len(node.args) <= 50):
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        raise ToolError(f"unsupported expression element: {type(node).__name__}")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"invalid expression: {exc.msg}") from exc
    try:
        result = ev(tree)
    except ZeroDivisionError as exc:
        raise ToolError("division by zero") from exc
    return float(result)


# ---------- tool specs (OpenAI function-calling shape) ----------
TOOL_SPECS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": (
                "Search the KMITL student regulations and guidebooks (undergraduate and graduate). "
                "Call this for any factual question about rules, grading, GPA, registration, leave, "
                "graduation, tuition, visas or student life at KMITL. Do NOT call it for greetings, "
                "thanks, or questions already answered earlier in the conversation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Focused search query in English"},
                    "k": {"type": "integer", "minimum": 1, "maximum": 8, "description": "Number of passages (default 5)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Evaluate an arithmetic expression, e.g. a GPA from grade points and credits: "
                "'(4*3 + 3.5*3 + 2*1) / (3+3+1)'. Supports + - * / // % ** and round, min, max, abs, sum."
            ),
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "Arithmetic only, max 200 chars"}},
                "required": ["expression"],
            },
        },
    },
]
ALLOWED_TOOLS = frozenset(spec["function"]["name"] for spec in TOOL_SPECS)


def validate_args(name: str, args: Any) -> dict:
    if name not in ALLOWED_TOOLS:
        raise ToolError(f"tool {name!r} is not allowed; available: {sorted(ALLOWED_TOOLS)}")
    if not isinstance(args, dict):
        raise ToolError("arguments must be a JSON object")
    if name == "search_docs":
        q = args.get("query")
        if not isinstance(q, str) or not q.strip():
            raise ToolError("search_docs needs a non-empty string 'query'")
        if len(q) > 300:
            raise ToolError("query too long (max 300 chars)")
        k = args.get("k", 5)
        if not isinstance(k, int) or isinstance(k, bool):
            k = 5
        return {"query": q.strip(), "k": max(1, min(8, k))}
    if name == "calculator":
        e = args.get("expression")
        if not isinstance(e, str) or not e.strip():
            raise ToolError("calculator needs a string 'expression'")
        if len(e) > 200:
            raise ToolError("expression too long (max 200 chars)")
        return {"expression": e.strip()}
    raise ToolError(f"no validator for {name!r}")  # pragma: no cover


def execute(name: str, args: dict, retriever: Retriever, min_score: float | None = None, expand_fn=None) -> tuple[Any, list]:
    """Run a validated tool. Returns (json-able result for the model, hits for citation bookkeeping)."""
    if name == "search_docs":
        hits = retriever.search(args["query"], k=args["k"], min_score=min_score, expand_fn=expand_fn)
        if not hits:
            return {"results": [], "note": "No passage passed the relevance floor. The documents may not cover this."}, []
        return {"results": [h.to_dict(max_chars=1200) for h in hits]}, hits
    if name == "calculator":
        return {"result": safe_calc(args["expression"])}, []
    raise ToolError(f"tool {name!r} is not allowed")
