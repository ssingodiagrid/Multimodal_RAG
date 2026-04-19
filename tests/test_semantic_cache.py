import tempfile
from pathlib import Path

from cache.semantic_cache import SemanticCache


def test_cache_miss_empty():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d) / "sc"
        c = SemanticCache(
            dim=4,
            base_path=str(base),
            threshold=0.99,
            max_entries=10,
        )
        hit = c.lookup([1.0, 0.0, 0.0, 0.0], fingerprint="fp1")
        assert hit is None


def test_cache_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d) / "sc"
        c = SemanticCache(dim=3, base_path=str(base), threshold=0.95, max_entries=10)
        qv = [0.6, 0.8, 0.0]
        n = sum(x * x for x in qv) ** 0.5
        qv = [x / n for x in qv]
        ctx = [
            {
                "chunk_id": "1",
                "text": "t",
                "page": 1,
                "score": 1.0,
                "score_source": "dense",
            }
        ]
        c.store(qv, fingerprint="fp1", query_text="q", answer="a", context=ctx)
        hit = c.lookup(qv, fingerprint="fp1")
        assert hit is not None
        assert hit.answer == "a"
        assert hit.context[0]["chunk_id"] == "1"


def test_cache_miss_wrong_fingerprint():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d) / "sc"
        c = SemanticCache(dim=2, base_path=str(base), threshold=0.5, max_entries=10)
        qv = [0.70710678, 0.70710678]
        c.store(qv, "fp-a", "q", "a", [])
        assert c.lookup(qv, "fp-b") is None


def test_cache_fifo_eviction():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d) / "sc"
        c = SemanticCache(dim=2, base_path=str(base), threshold=0.9, max_entries=2)
        c.store([1.0, 0.0], "fp", "q1", "a1", [])
        c.store([0.0, 1.0], "fp", "q2", "a2", [])
        c.store([0.707, 0.707], "fp", "q3", "a3", [])
        assert c.lookup([1.0, 0.0], "fp") is None
        hit = c.lookup([0.0, 1.0], "fp")
        assert hit is not None and hit.answer == "a2"
