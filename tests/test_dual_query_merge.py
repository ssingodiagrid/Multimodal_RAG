from retrieval.dual_query_merge import merge_dual_retrieval_contexts


def _chunk(cid: str, score: float, text: str = "t") -> dict:
    return {"chunk_id": cid, "score": score, "text": text, "page": 1}


def test_merge_union_dedupes_prefers_higher_score_row():
    raw = [_chunk("a", 0.5), _chunk("b", 0.9)]
    ref = [_chunk("a", 0.8, "better"), _chunk("c", 0.3)]
    out = merge_dual_retrieval_contexts(raw, ref, top_k=10)
    by = {c["chunk_id"]: c for c in out}
    assert set(by) == {"a", "b", "c"}
    assert by["a"]["score"] == 0.8
    assert by["a"]["text"] == "better"
    assert by["a"]["retrieval_source"] == "both"
    assert by["b"]["retrieval_source"] == "raw"
    assert by["c"]["retrieval_source"] == "refined"


def test_merge_respects_top_k_order():
    raw = [_chunk("x", 0.1)]
    ref = [_chunk("y", 0.5), _chunk("z", 0.9)]
    out = merge_dual_retrieval_contexts(raw, ref, top_k=2)
    assert [c["chunk_id"] for c in out] == ["z", "y"]


def test_merge_empty_refined():
    raw = [_chunk("a", 0.5)]
    out = merge_dual_retrieval_contexts(raw, [], 5)
    assert len(out) == 1
    assert out[0]["chunk_id"] == "a"
    assert out[0]["retrieval_source"] == "raw"
