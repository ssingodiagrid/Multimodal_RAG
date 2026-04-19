from ingestion.chunker import chunk_pages
from ingestion.text_cleaner import clean_text


def test_clean_text_collapses_whitespace():
    raw = "Revenue   increased\n\n\nin FY24."
    cleaned = clean_text(raw)
    assert cleaned == "Revenue increased in FY24."


def test_chunk_pages_returns_metadata():
    pages = [{"page": 1, "text": "A " * 500}]
    chunks = chunk_pages("doc_1", pages, chunk_size=120, chunk_overlap=20)
    assert len(chunks) > 1
    assert chunks[0]["doc_id"] == "doc_1"
    assert "chunk_id" in chunks[0]
    assert chunks[0]["page"] == 1


def test_chunk_overlap_not_zero():
    pages = [{"page": 1, "text": "x" * 2000}]
    chunks = chunk_pages("doc", pages, chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 1
