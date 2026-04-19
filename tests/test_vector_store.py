from pathlib import Path
import tempfile

from configs.settings import Settings
from retrieval.vector_store import FaissVectorStore, InMemoryVectorStore


def test_settings_defaults():
    s = Settings()
    assert s.chunk_size == 800
    assert s.chunk_overlap == 100
    assert s.top_k == 5


def test_add_and_search_vectors():
    store = InMemoryVectorStore()
    store.add([[1.0, 0.0], [0.0, 1.0]], [{"chunk_id": "a"}, {"chunk_id": "b"}])
    out = store.search([1.0, 0.0], top_k=1)
    assert out[0]["metadata"]["chunk_id"] == "a"


def test_faiss_save_load_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        base = str(Path(tmp) / "idx")
        store = FaissVectorStore(dim=2)
        store.add([[1.0, 0.0], [0.0, 1.0]], [{"chunk_id": "a"}, {"chunk_id": "b"}])
        store.save(base)
        loaded = FaissVectorStore.load(base)
        out = loaded.search([1.0, 0.0], top_k=1)
        assert out[0]["metadata"]["chunk_id"] == "a"
