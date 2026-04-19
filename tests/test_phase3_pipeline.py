from configs.settings import Settings
from retrieval.pipeline import build_final_context, safe_mode_from_flags


def test_phase3_settings_defaults_exist():
    s = Settings()
    assert hasattr(s, "hybrid_alpha")
    assert hasattr(s, "enable_hybrid")
    assert hasattr(s, "enable_rerank")


def test_build_final_context_schema():
    candidates = [{"chunk_id": "c1", "text": "abc", "page": 1, "hybrid_score": 0.8}]
    out = build_final_context(candidates, top_k=1)
    assert out == [
        {
            "chunk_id": "c1",
            "text": "abc",
            "page": 1,
            "score": 0.8,
            "score_source": "hybrid",
        }
    ]


def test_build_final_context_score_source_dense_and_rerank():
    dense = [{"chunk_id": "d", "text": "x", "page": 0, "score": 0.5}]
    assert build_final_context(dense, top_k=1)[0]["score_source"] == "dense"
    rerank = [
        {
            "chunk_id": "r",
            "text": "y",
            "page": 1,
            "hybrid_score": 0.9,
            "rerank_score": -2.3,
        }
    ]
    row = build_final_context(rerank, top_k=1)[0]
    assert row["score"] == -2.3
    assert row["score_source"] == "rerank"


def test_pipeline_dense_only_fallback():
    mode = safe_mode_from_flags(enable_hybrid=False, enable_rerank=False)
    assert mode == "dense_only"

