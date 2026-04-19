def _token_overlap(a: str, b: str) -> float:
    sa = set((a or "").lower().split())
    sb = set((b or "").lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(len(sa), 1)


def compute_basic_ragas_like_metrics(
    query: str,
    expected_answer: str,
    model_answer: str,
    retrieved_context: list[dict],
) -> dict:
    context_text = " ".join([c.get("text", "") for c in retrieved_context])
    return {
        "answer_relevance": _token_overlap(expected_answer, model_answer),
        "faithfulness": _token_overlap(model_answer, context_text),
        "context_relevance": _token_overlap(query, context_text),
    }

