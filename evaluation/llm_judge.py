import json

from generation.llm_pipeline import GeminiClient


def build_judge_prompt(query: str, expected: str, predicted: str) -> str:
    return f"""
You are an impartial evaluator.
Score:
1) correctness (1-5)
2) hallucination (yes/no)
3) reasoning quality (1-5)

Return strict JSON with keys:
correctness, hallucination, reasoning_quality, rationale

Query: {query}
Expected answer: {expected}
Predicted answer: {predicted}
""".strip()


def judge_with_llm(query: str, expected: str, predicted: str) -> dict:
    prompt = build_judge_prompt(query, expected, predicted)
    raw = GeminiClient().answer(prompt)
    try:
        return json.loads(raw)
    except Exception:
        return {
            "correctness": None,
            "hallucination": None,
            "reasoning_quality": None,
            "rationale": raw,
        }

