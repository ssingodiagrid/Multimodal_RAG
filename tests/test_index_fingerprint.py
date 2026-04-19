import tempfile
from pathlib import Path

from cache.index_fingerprint import compute_index_fingerprint


def test_fingerprint_changes_when_file_changes():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "fake.index"
        faiss_f = Path(str(p) + ".faiss")
        faiss_f.write_bytes(b"a")
        fp1 = compute_index_fingerprint(str(p), sparse_path=None, embedding_model="m1")
        faiss_f.write_bytes(b"ab")
        fp2 = compute_index_fingerprint(str(p), sparse_path=None, embedding_model="m1")
        assert fp1 != fp2


def test_fingerprint_includes_embedding_model():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "fake.index"
        Path(str(p) + ".faiss").write_bytes(b"x")
        a = compute_index_fingerprint(str(p), None, "model-a")
        b = compute_index_fingerprint(str(p), None, "model-b")
        assert a != b


def test_fingerprint_changes_with_visual_index():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "text.index"
        Path(str(p) + ".faiss").write_bytes(b"x")
        v = Path(d) / "vis.index"
        Path(str(v) + ".faiss").write_bytes(b"v")
        a = compute_index_fingerprint(
            str(p), None, "m", visual_faiss_path=None, visual_embedding_model=None
        )
        b = compute_index_fingerprint(
            str(p), None, "m", visual_faiss_path=str(v), visual_embedding_model="clip-x"
        )
        assert a != b
