from retrieval.bm25 import SimpleBM25Retriever


def test_bm25_returns_relevant_chunk():
    chunks = [
        {"chunk_id": "c1", "text": "net income increased in fy24", "page": 1, "doc_id": "d"},
        {"chunk_id": "c2", "text": "supply chain delays impacted costs", "page": 2, "doc_id": "d"},
    ]
    r = SimpleBM25Retriever(chunks)
    out = r.search("net income", top_k=1)
    assert out[0]["chunk_id"] == "c1"


def test_bm25_can_initialize_from_chunk_list():
    chunks = [{"chunk_id": "c1", "text": "a b c", "page": 1, "doc_id": "d"}]
    r = SimpleBM25Retriever(chunks)
    assert r is not None

