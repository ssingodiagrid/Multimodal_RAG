from retrieval.pipeline import build_final_context


def test_build_final_context_preserves_modality_fields():
    c = {
        "chunk_id": "1",
        "text": "t",
        "page": 2,
        "hybrid_score": 0.5,
        "modality": "table",
        "table_json": {"headers": ["h"], "rows": [["v"]]},
        "doc_id": "d",
    }
    out = build_final_context([c], top_k=1)[0]
    assert out["modality"] == "table"
    assert out["table_json"]["headers"] == ["h"]
    assert out["doc_id"] == "d"
