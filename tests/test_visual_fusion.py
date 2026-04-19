from retrieval.visual_fusion import merge_text_and_visual_candidates


def test_merge_text_only_unchanged_order_when_no_visual_overlap():
    text = [
        {"chunk_id": "a", "text": "x", "page": 1, "score": 0.9},
        {"chunk_id": "b", "text": "y", "page": 1, "score": 0.1},
    ]
    out = merge_text_and_visual_candidates(text, [], 0.65)
    assert [r["chunk_id"] for r in out] == ["a", "b"]


def test_merge_visual_boosts_shared_chunk():
    text = [
        {"chunk_id": "a", "text": "low", "page": 1, "score": 0.5},
        {"chunk_id": "b", "text": "high text", "page": 1, "score": 0.5},
    ]
    visual = [
        {"chunk_id": "a", "text": "l", "page": 1, "score": 0.99, "metadata": {}},
        {"chunk_id": "b", "text": "h", "page": 1, "score": 0.01, "metadata": {}},
    ]
    out = merge_text_and_visual_candidates(text, visual, 0.65)
    assert out[0]["chunk_id"] == "a"


def test_merge_adds_visual_only_chunk():
    text = [{"chunk_id": "b", "text": "only", "page": 1, "score": 0.5}]
    visual = [
        {
            "chunk_id": "img1",
            "text": "caption",
            "page": 2,
            "score": 0.95,
            "metadata": {"chunk_id": "img1", "modality": "image"},
        }
    ]
    out = merge_text_and_visual_candidates(text, visual, 0.65)
    ids = {r["chunk_id"] for r in out}
    assert ids == {"b", "img1"}
