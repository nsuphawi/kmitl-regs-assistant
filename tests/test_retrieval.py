from app.bm25 import tokenize
from app.retrieve import rrf
from evaluation.dataset import relevant_ids
from evaluation.metrics import hit_at_k, precision_at_k, recall_at_k, reciprocal_rank


def test_metrics():
    assert recall_at_k(["a", "b", "c"], ["b", "z"], 2) == 0.5
    assert recall_at_k(["a"], [], 3) == 0.0
    assert precision_at_k(["a", "b"], {"a"}, 2) == 0.5
    assert reciprocal_rank(["x", "y", "a"], {"a"}) == 1 / 3
    assert reciprocal_rank(["x"], {"a"}) == 0.0
    assert hit_at_k(["x", "a"], {"a", "b"}, 2) == 1.0


def test_rrf_fuses_by_rank_and_is_stable():
    assert [i for i, _ in rrf([[1, 2, 3], [3, 1]])] == [1, 3, 2]
    assert [i for i, _ in rrf([[7], [8]])] == [7, 8]  # tie -> first appearance


def test_tokenize_keeps_codes():
    toks = tokenize("Submit form ACAD-07 or TM.47 within 90 days")
    assert "acad-07" in toks and "tm.47" in toks and "90" in toks


def test_bm25_finds_exact_rare_token(retriever):
    hits = retriever.search("ACAD-07", k=3, mode="bm25", min_score=-1)
    assert hits and "ACAD-07" in hits[0].chunk.text


def test_hybrid_returns_relevant_chunk(retriever):
    hits = retriever.search("How many credits can I take in a normal semester?", k=3, mode="hybrid", min_score=-1)
    assert any("22 credits" in h.chunk.text for h in hits)


def test_min_score_floor_drops_everything_for_off_topic(retriever):
    assert retriever.search("quantum chromodynamics lattice", k=5, min_score=0.2) == []


def test_chunks_track_pages_and_ids(index):
    ids = [c.id for c in index.chunks]
    assert len(ids) == len(set(ids)) and all(c.page_start >= 1 for c in index.chunks)


def test_gold_matching_by_evidence_quote(index):
    item = {"id": "t", "question": "?", "answerable": True,
            "evidence": ["shall be dismissed"], "doc": "example_handbook", "doc_only": True}
    rel = relevant_ids(item, index.chunks)
    assert rel and all("dismissed" in c.text for c in index.chunks if c.id in rel)
    assert relevant_ids({**item, "answerable": False}, index.chunks) == set()


def test_hyde_expansion_bridges_vocabulary(retriever):
    q = "What happens to students with a really bad average?"
    gold = lambda hits: any("dismissed" in h.chunk.text for h in hits[:1])
    expand = lambda _: "A student whose cumulative grade point average falls below 1.50 shall be dismissed."
    assert gold(retriever.search(q, k=3, min_score=-1, expand_fn=expand))


def test_hyde_failure_falls_back_to_plain_query(retriever):
    plain = retriever.search("ACAD-07", k=3, min_score=-1)
    assert [h.chunk.id for h in retriever.search("ACAD-07", k=3, min_score=-1, expand_fn=lambda _: "")] == [h.chunk.id for h in plain]


def test_clause_chunking_isolates_numbered_clause():
    from app.ingest import Page, chunk_pages, clause_starts
    text = ("Article 14. Enrollment and payment of tuition fees: 14.1 Students must enroll and pay fees. "
            + "word " * 120 + "end. 14.4 In each regular semester, students have to enroll in not less than "
            "9 credits and not more than 22 credits. Article 15. Late payment under Article 42.4 hereof is fined.")
    words = text.split()
    starts = clause_starts(words)
    assert words.index("14.4") in starts and words.index("15.") - 1 in starts
    assert words.index("42.4") not in starts  # cross-reference, not a boundary
    chunks = chunk_pages([Page("d", "D", 1, text)], chunk_words=60, overlap=10, strategy="clause")
    target = [c for c in chunks if "not more than 22 credits" in c.text]
    assert len(target) == 1 and target[0].text.startswith("[Article 14. Enrollment")
    assert "word word" not in target[0].text  # not diluted by the long 14.1 clause
