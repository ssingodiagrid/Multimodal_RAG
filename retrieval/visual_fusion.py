"""Fuse Phase 3 text-path scores with visual FAISS scores (Phase 6)."""

from __future__ import annotations


def _minmax_values(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return {k: 0.5 for k in scores}
    return {k: (scores[k] - lo) / (hi - lo) for k in scores}


def _primary_text_score(c: dict) -> float:
    if "rerank_score" in c:
        return float(c["rerank_score"])
    if "hybrid_score" in c:
        return float(c["hybrid_score"])
    return float(c.get("score", 0.0))


def merge_text_and_visual_candidates(
    text_cands: list[dict],
    visual_cands: list[dict],
    lambda_text: float,
) -> list[dict]:
    """
    Union by chunk_id. Fused score = λ * norm(text_score) + (1-λ) * norm(visual_score).
    Text-only: fused = norm(text). Visual-only: fused = (1-λ) * norm(visual).
    """
    lt = max(0.0, min(1.0, float(lambda_text)))
    text_scores = {c["chunk_id"]: _primary_text_score(c) for c in text_cands}
    vis_scores = {c["chunk_id"]: float(c["score"]) for c in visual_cands}
    nt = _minmax_values(text_scores)
    nv = _minmax_values(vis_scores)

    by_cid: dict[str, dict] = {c["chunk_id"]: dict(c) for c in text_cands}
    for vc in visual_cands:
        cid = vc["chunk_id"]
        if cid not in by_cid:
            meta = vc.get("metadata")
            if not isinstance(meta, dict):
                meta = {}
            by_cid[cid] = {
                "chunk_id": cid,
                "text": vc.get("text", meta.get("text", "")),
                "page": vc.get("page", meta.get("page", -1)),
                "doc_id": vc.get("doc_id", meta.get("doc_id", "")),
                "score": float(vc.get("score", 0.0)),
                "metadata": meta if meta else {},
            }

    fused_rows: list[dict] = []
    for cid, row in by_cid.items():
        row = dict(row)
        t_norm = nt.get(cid)
        v_norm = nv.get(cid)
        if t_norm is not None and v_norm is not None:
            fused = lt * t_norm + (1.0 - lt) * v_norm
        elif t_norm is not None:
            fused = t_norm
        elif v_norm is not None:
            fused = (1.0 - lt) * v_norm
        else:
            fused = 0.0
        row.pop("rerank_score", None)
        row["hybrid_score"] = float(fused)
        row["score"] = float(fused)
        fused_rows.append(row)

    fused_rows.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return fused_rows
