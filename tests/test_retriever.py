from generation.prompt_builder import build_grounded_prompt
from retrieval.retriever import retrieve_context


def test_retrieve_context_returns_ranked_chunks():
    store_results = [
        {"score": 0.9, "metadata": {"chunk_id": "c1", "text": "Revenue grew", "page": 1}},
        {"score": 0.8, "metadata": {"chunk_id": "c2", "text": "Costs rose", "page": 2}},
    ]
    context = retrieve_context(store_results, top_k=1)
    assert len(context) == 1
    assert context[0]["chunk_id"] == "c1"


def test_prompt_contains_query_and_context():
    p = build_grounded_prompt(
        "What changed?",
        [{"text": "Net income increased", "page": 3, "chunk_id": "x"}],
    )
    assert "What changed?" in p
    assert "(p" in p or "[p3]" in p


def test_grounding_instruction_present():
    p = build_grounded_prompt("Q", [{"text": "A", "page": 1, "chunk_id": "1"}])
    assert "Use only the provided context" in p
