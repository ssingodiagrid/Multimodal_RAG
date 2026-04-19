"""Merge retrieval results from a raw user query and an LLM-refined query."""

from __future__ import annotations

from collections import defaultdict


def merge_dual_retrieval_contexts(
    raw_ctx: list[dict],
    refined_ctx: list[dict],
    top_k: int,
) -> list[dict]:
    """
    Union by chunk_id. Combined score is max(raw, refined); the row kept is from
    whichever pass had the higher score for that id. Adds ``retrieval_source``:
    ``raw``, ``refined``, or ``both``.
    """
    best_score: dict[str, float] = {}
    best_row: dict[str, dict] = {}
    sources: dict[str, set[str]] = defaultdict(set)

    def consider(lst: list[dict], label: str) -> None:
        for c in lst:
            cid = c.get("chunk_id")
            if cid is None:
                continue
            key = str(cid)
            s = float(c.get("score", 0.0))
            sources[key].add(label)
            if key not in best_score or s > best_score[key]:
                best_score[key] = s
                row = dict(c)
                row["score"] = s
                best_row[key] = row

    consider(raw_ctx, "raw")
    consider(refined_ctx, "refined")

    out: list[dict] = []
    for key, row in best_row.items():
        r = dict(row)
        r["score"] = best_score[key]
        src = sources[key]
        if len(src) > 1:
            r["retrieval_source"] = "both"
        elif "raw" in src:
            r["retrieval_source"] = "raw"
        else:
            r["retrieval_source"] = "refined"
        out.append(r)
    out.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    cap = max(0, int(top_k))
    return out[:cap]
