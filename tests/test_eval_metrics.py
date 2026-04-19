from evaluation.eval_runner import build_eval_sample, evaluate_row
from evaluation.llm_judge import build_judge_prompt
from evaluation.ragas_eval import compute_basic_ragas_like_metrics


def test_build_eval_sample_shape():
    sample = build_eval_sample(
        query="What changed?",
        expected="Revenue increased",
        answer="Revenue increased in FY24.",
        context=[{"text": "Revenue increased in FY24.", "page": 2}],
    )
    assert sample["query"] == "What changed?"
    assert "retrieved_context" in sample
    assert sample["model_answer"].startswith("Revenue")


def test_basic_ragas_like_metrics_keys():
    m = compute_basic_ragas_like_metrics(
        query="q",
        expected_answer="revenue increased",
        model_answer="revenue increased in fy24",
        retrieved_context=[{"text": "revenue increased in fy24"}],
    )
    assert "answer_relevance" in m
    assert "faithfulness" in m
    assert "context_relevance" in m


def test_build_judge_prompt_contains_axes():
    p = build_judge_prompt("q", "gold", "pred")
    assert "correctness" in p.lower()
    assert "hallucination" in p.lower()
    assert "reasoning quality" in p.lower()


def test_evaluate_row_has_ragas_and_judge(monkeypatch):
    monkeypatch.setattr(
        "evaluation.eval_runner.judge_with_llm", lambda *a, **k: {"correctness": 4}
    )
    row = evaluate_row(
        {"query": "q", "expected_answer": "a"},
        answer="a",
        context=[{"text": "a"}],
    )
    assert "ragas" in row
    assert "judge" in row

