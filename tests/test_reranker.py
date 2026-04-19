from reranking.cross_encoder import rerank_candidates


def test_rerank_candidates_shape():
    candidates = [
        {"chunk_id": "c1", "text": "revenue growth in fy24", "score": 0.4},
        {"chunk_id": "c2", "text": "supply chain issue", "score": 0.6},
    ]
    out = rerank_candidates("revenue growth", candidates, top_k=1, model=None)
    assert len(out) == 1
    assert "chunk_id" in out[0]


def test_rerank_candidates_no_truncate_returns_all_without_model():
    candidates = [
        {"chunk_id": "c1", "text": "a", "score": 0.1},
        {"chunk_id": "c2", "text": "b", "score": 0.2},
    ]
    out = rerank_candidates(
        "q", candidates, top_k=1, model=None, truncate=False
    )
    assert len(out) == 2

