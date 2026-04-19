from retrieval.hybrid_retriever import fuse_dense_sparse


def test_fuse_dense_sparse_merges_scores():
    dense = [{"chunk_id": "c1", "score": 0.9}, {"chunk_id": "c2", "score": 0.3}]
    sparse = [{"chunk_id": "c2", "score": 2.0}, {"chunk_id": "c3", "score": 1.0}]
    out = fuse_dense_sparse(dense, sparse, alpha=0.7, top_n=3)
    ids = [o["chunk_id"] for o in out]
    assert "c1" in ids and "c2" in ids and "c3" in ids

