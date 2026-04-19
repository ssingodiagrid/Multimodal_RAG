from __future__ import annotations

from retrieval.modality_router import ModalityIntent


def candidate_modality(c: dict) -> str:
    m = c.get("metadata")
    if isinstance(m, dict) and m.get("modality"):
        return str(m["modality"])
    return str(c.get("modality") or "text")


def apply_modality_preference(
    candidates: list[dict],
    intent: ModalityIntent | None,
    enabled: bool,
) -> list[dict]:
    """Prefer chunks whose modality matches router intent; stable tie-break by score."""
    if not candidates or not enabled or not intent or intent == "mixed":
        return candidates

    def score_tuple(c: dict) -> tuple[int, float]:
        mod = candidate_modality(c)
        match = 1 if mod == intent else 0
        s = float(
            c.get("rerank_score", c.get("hybrid_score", c.get("score", 0.0)))
        )
        return (match, s)

    return sorted(candidates, key=score_tuple, reverse=True)
