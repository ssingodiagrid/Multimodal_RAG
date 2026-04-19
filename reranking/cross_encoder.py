from __future__ import annotations

from typing import Any


def rerank_candidates(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    model: Any = None,
    *,
    truncate: bool = True,
) -> list[dict]:
    if not candidates:
        return []
    if model is None:
        out = [dict(c) for c in candidates]
        if truncate:
            return out[:top_k]
        return out
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)
    rescored = []
    for cand, score in zip(candidates, scores):
        item = dict(cand)
        item["rerank_score"] = float(score)
        rescored.append(item)
    rescored.sort(key=lambda x: x["rerank_score"], reverse=True)
    if truncate:
        return rescored[:top_k]
    return rescored

