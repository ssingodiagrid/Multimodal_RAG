from evaluation.llm_judge import judge_with_llm
from evaluation.ragas_eval import compute_basic_ragas_like_metrics


def build_eval_sample(query: str, expected: str, answer: str, context: list[dict]) -> dict:
    return {
        "query": query,
        "expected_answer": expected,
        "retrieved_context": context,
        "model_answer": answer,
    }


def evaluate_row(row: dict, answer: str, context: list[dict]) -> dict:
    ragas = compute_basic_ragas_like_metrics(
        query=row["query"],
        expected_answer=row["expected_answer"],
        model_answer=answer,
        retrieved_context=context,
    )
    try:
        judge = judge_with_llm(row["query"], row["expected_answer"], answer)
    except Exception as exc:
        judge = {
            "correctness": None,
            "hallucination": None,
            "reasoning_quality": None,
            "rationale": f"judge_error: {exc}",
        }
    out = dict(row)
    out["model_answer"] = answer
    out["retrieved_context"] = context
    out["ragas"] = ragas
    out["judge"] = judge
    return out

