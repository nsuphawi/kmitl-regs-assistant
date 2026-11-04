import pytest

from app.agent import parse_tool_call, run_agent
from app.answer import answer_rag, validate_answer
from app.tools import ToolError, safe_calc, validate_args


# ---------------- tools ----------------
def test_calculator_gpa():
    assert safe_calc("round((4*3 + 3.5*3 + 2*1) / 7, 2)") == 3.5


@pytest.mark.parametrize("expr", ["__import__('os').system('ls')", "open('x')", "2**999999", "a+1", "(lambda: 1)()"])
def test_calculator_rejects_unsafe(expr):
    with pytest.raises(ToolError):
        safe_calc(expr)


def test_validate_args_allow_list_and_clamping():
    with pytest.raises(ToolError):
        validate_args("delete_everything", {})
    with pytest.raises(ToolError):
        validate_args("search_docs", {"query": ""})
    assert validate_args("search_docs", {"query": " gpa ", "k": 99}) == {"query": "gpa", "k": 8}
    assert validate_args("search_docs", {"query": "gpa", "k": "x"})["k"] == 5


def test_parse_tool_call_variants():
    assert parse_tool_call('```json\n{"tool": "calculator", "args": {"expression": "1+1"}}\n```')["tool"] == "calculator"
    assert parse_tool_call('{"tool": "x", "args": [1]}') == {"tool": "x", "args": {}}
    assert parse_tool_call("Hello there") == {"answer": "Hello there"}


# ---------------- structured output ----------------
def test_validate_answer_drops_unknown_citations():
    raw = '{"answer": "22", "citations": [{"chunk_id": "a:1", "quote": "x"}, {"chunk_id": "evil", "quote": "y"}]}'
    ans, dropped = validate_answer(raw, {"a:1"})
    assert [c.chunk_id for c in ans.citations] == ["a:1"] and dropped == ["evil"]
    with pytest.raises(ValueError):
        validate_answer('{"answer": "22", "citations": []}', {"a:1"})


def test_rag_repairs_once_then_succeeds(retriever, fake_llm):
    hits = retriever.search("maximum credits per regular semester", k=3, min_score=-1)
    good = '{"answer": "22 credits", "citations": [{"chunk_id": "%s", "quote": "22 credits"}], "abstained": false}' % hits[0].chunk.id
    llm = fake_llm(["not json at all", good])
    res = answer_rag("maximum credits per regular semester", retriever, k=3, min_score=-1, chat_fn=llm)
    assert res.repaired and res.answer.answer == "22 credits" and len(llm.calls) == 2


def test_rag_abstains_without_model_call_when_floor_blocks(retriever, fake_llm):
    llm = fake_llm([])
    res = answer_rag("quantum chromodynamics lattice", retriever, min_score=0.5, chat_fn=llm)
    assert res.answer.abstained and llm.calls == []


# ---------------- agent loop ----------------
def _first_hit_id(retriever, q):
    return retriever.search(q, k=5, min_score=-1)[0].chunk.id


def test_agent_search_then_answer(retriever, fake_llm):
    cid = _first_hit_id(retriever, "leave of absence semesters")
    llm = fake_llm([
        [("search_docs", {"query": "leave of absence semesters"})],
        '{"answer": "Two consecutive semesters.", "citations": [{"chunk_id": "%s", "quote": "two consecutive"}], "abstained": false}' % cid,
    ])
    res = run_agent("How long can I take leave?", retriever, chat_fn=llm, min_score=-1)
    assert res.stopped == "answer" and res.steps == 2
    assert [c.chunk_id for c in res.answer.citations] == [cid]
    assert any(e["type"] == "tool_result" and e["ok"] for e in res.trace)


def test_agent_answers_chitchat_without_tools(retriever, fake_llm):
    res = run_agent("thanks!", retriever, chat_fn=fake_llm(["You're welcome!"]))
    assert res.answer.answer == "You're welcome!" and res.steps == 1 and not res.answer.citations


def test_agent_rejects_unknown_tool_and_caps_steps(retriever, fake_llm):
    llm = fake_llm([[("rm_rf", {})]] * 3 + ['{"answer": "I could not find this.", "citations": [], "abstained": true}'])
    res = run_agent("q", retriever, chat_fn=llm, max_steps=3)
    assert res.stopped == "max_steps" and res.answer.abstained
    assert all(not e["ok"] for e in res.trace if e["type"] == "tool_result")


def test_agent_repeat_query_guard(retriever, fake_llm):
    call = [("search_docs", {"query": "visa report"})]
    llm = fake_llm([call, call, '{"answer": "x", "citations": [], "abstained": true}'])
    res = run_agent("visa?", retriever, chat_fn=llm)
    results = [e for e in res.trace if e["type"] == "tool_result"]
    assert "Already called" in results[1]["summary"]


def test_agent_falls_back_to_prompted_json(retriever, fake_llm):
    llm = fake_llm(['{"tool": "calculator", "args": {"expression": "2*3"}}', "The answer is 6."], reject_tools=True)
    res = run_agent("what is 2*3", retriever, chat_fn=llm)
    assert res.answer.answer == "The answer is 6."
    assert any(e["type"] == "info" for e in res.trace)
    assert "TOOL RESULT for calculator" in llm.calls[-1]["messages"][-1]["content"]


def test_agent_hallucinated_citation_triggers_repair(retriever, fake_llm):
    cid = _first_hit_id(retriever, "visa 90 days")
    llm = fake_llm([
        [("search_docs", {"query": "visa 90 days"})],
        '{"answer": "Every 90 days.", "citations": [{"chunk_id": "made-up", "quote": ""}], "abstained": false}',
        '{"answer": "Every 90 days.", "citations": [{"chunk_id": "%s", "quote": "90 days"}], "abstained": false}' % cid,
    ])
    res = run_agent("visa reporting?", retriever, chat_fn=llm, min_score=-1)
    assert any(e["type"] == "repair" for e in res.trace) and res.answer.citations[0].chunk_id == cid


def test_validation_tolerates_common_small_model_quirks():
    allowed = {"doc:0013"}
    # raw newline inside a JSON string + bracketed id + over-long quote
    raw = '{"answer": "Two plans:\n* Plan 1: 72 credits", "citations": [{"chunk_id": "[doc:0013]", "quote": "%s"}]}' % ("x" * 900)
    ans, _ = validate_answer(raw, allowed)
    assert ans.citations[0].chunk_id == "doc:0013" and len(ans.citations[0].quote) == 400
    # citations given as plain strings
    ans, _ = validate_answer('{"answer": "a", "citations": ["doc:0013"]}', allowed)
    assert ans.citations[0].chunk_id == "doc:0013"
    # empty citations but the id is cited inline in the answer text
    ans, _ = validate_answer('{"answer": "9 to 22 credits [doc:0013].", "citations": []}', allowed)
    assert [c.chunk_id for c in ans.citations] == ["doc:0013"]
    # an inline id that was NOT in the context is still rejected
    with pytest.raises(ValueError):
        validate_answer('{"answer": "made up [doc:9999].", "citations": []}', allowed)
