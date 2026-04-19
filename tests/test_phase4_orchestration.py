import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from configs.settings import Settings
from main import run_rag_query
from retrieval.vector_store import FaissVectorStore


def test_multihop_calls_retrieval_twice():
    settings = Settings()
    settings.enable_semantic_cache = False
    settings.enable_multi_hop = True
    settings.multi_hop_mode = "always"
    settings.multi_hop_merged_top_k = 5

    store = FaissVectorStore(2)
    store.add([[1.0, 0.0]], [{"chunk_id": "0", "text": "x", "page": 1, "doc_id": "d"}])

    embedder = MagicMock()
    embedder.embed_query.side_effect = lambda q: [0.5, 0.5]

    calls: list[str] = []

    def fake_phase3(q: str, qv, st, sett, modality_intent=None, **kwargs):
        calls.append(q)
        if q == "main question":
            return [
                {
                    "chunk_id": "1",
                    "text": "a",
                    "page": 1,
                    "score": 0.9,
                    "score_source": "dense",
                }
            ]
        return [
            {
                "chunk_id": "2",
                "text": "b",
                "page": 2,
                "score": 0.8,
                "score_source": "dense",
            }
        ]

    with patch("main.run_phase3_retrieval", side_effect=fake_phase3), patch(
        "main.generate_sub_query", return_value="hop two query"
    ), patch("main.answer_query", return_value="merged answer") as ans:
        a, ctx = run_rag_query("main question", store, embedder, settings)

    assert a == "merged answer"
    assert len(calls) == 2
    assert calls[0] == "main question"
    assert calls[1] == "hop two query"
    assert len(ctx) == 2
    assert {c["chunk_id"] for c in ctx} == {"1", "2"}
    ans.assert_called_once()


def test_cache_hit_skips_retrieval():
    settings = Settings()
    settings.enable_semantic_cache = True
    settings.enable_multi_hop = False

    store = FaissVectorStore(2)
    embedder = MagicMock()
    embedder.embed_query.return_value = [1.0, 0.0]

    with tempfile.TemporaryDirectory() as tmp:
        settings.semantic_cache_path = str(Path(tmp) / "sc")
        from cache.semantic_cache import SemanticCache

        SemanticCache(2, settings.semantic_cache_path, 0.9, 10).store(
            [1.0, 0.0],
            fingerprint="fp",
            query_text="q",
            answer="cached",
            context=[
                {
                    "chunk_id": "z",
                    "text": "t",
                    "page": 1,
                    "score": 1.0,
                    "score_source": "dense",
                }
            ],
        )

        with patch("main.compute_index_fingerprint", return_value="fp"), patch(
            "main.run_phase3_retrieval"
        ) as p3:
            a, ctx = run_rag_query("q", store, embedder, settings)

    assert a == "cached"
    assert ctx[0]["chunk_id"] == "z"
    p3.assert_not_called()
