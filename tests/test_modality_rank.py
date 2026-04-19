from retrieval.modality_rank import apply_modality_preference, candidate_modality


def test_candidate_modality_from_metadata():
    c = {"metadata": {"modality": "table"}, "chunk_id": "x"}
    assert candidate_modality(c) == "table"


def test_apply_modality_prefers_table():
    c = [
        {"chunk_id": "a", "text": "x", "page": 1, "hybrid_score": 0.9, "modality": "text"},
        {"chunk_id": "b", "text": "y", "page": 1, "hybrid_score": 0.5, "modality": "table"},
    ]
    out = apply_modality_preference(c, "table", True)
    assert out[0]["chunk_id"] == "b"


def test_apply_disabled_returns_same_order():
    c = [
        {"chunk_id": "a", "hybrid_score": 0.9, "modality": "text"},
        {"chunk_id": "b", "hybrid_score": 0.5, "modality": "table"},
    ]
    out = apply_modality_preference(c, "table", False)
    assert [x["chunk_id"] for x in out] == ["a", "b"]


def test_apply_image_intent_with_router_off_heuristic_simulation():
    """Pipeline passes enabled=True with rank_intent=image so figure chunks surface."""
    c = [
        {"chunk_id": "t1", "rerank_score": 0.9, "modality": "text"},
        {"chunk_id": "i1", "rerank_score": 0.1, "modality": "image"},
    ]
    out = apply_modality_preference(c, "image", True)
    assert out[0]["chunk_id"] == "i1"
