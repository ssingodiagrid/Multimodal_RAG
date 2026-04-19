from __future__ import annotations

from retrieval.bm25 import SimpleBM25Retriever, load_sparse_corpus
from retrieval.hybrid_retriever import fuse_dense_sparse
from retrieval.modality_rank import apply_modality_preference
from retrieval.visual_fusion import merge_text_and_visual_candidates
from retrieval.visual_index import should_run_visual_merge, visual_merge_gate_intent
from reranking.cross_encoder import rerank_candidates

_OPTIONAL_CTX = (
    "modality",
    "asset_path",
    "table_json",
    "table_id",
    "image_id",
    "doc_id",
    "pdf_caption",
    "colpali_rank",
)


def _extras_from_candidate(c: dict) -> dict:
    out: dict = {}
    meta = c.get("metadata") if isinstance(c.get("metadata"), dict) else {}
    for k in _OPTIONAL_CTX:
        v = c.get(k)
        if v is None and k in meta:
            v = meta[k]
        if v is not None:
            out[k] = v
    return out


def safe_mode_from_flags(enable_hybrid: bool, enable_rerank: bool) -> str:
    if not enable_hybrid:
        return "dense_only"
    if enable_hybrid and not enable_rerank:
        return "hybrid_no_rerank"
    return "hybrid_rerank"


def _rank_intent_and_preference_enabled(
    modality_intent: str | None,
    router_on: bool,
    visual_gate: str | None,
) -> tuple[str | None, bool]:
    """Order chunks by modality when router is on, or by heuristic gate when router is off."""
    rank_intent = modality_intent if router_on else visual_gate
    prefer = router_on or (rank_intent is not None and rank_intent != "mixed")
    return rank_intent, prefer


def build_final_context(candidates: list[dict], top_k: int = 5) -> list[dict]:
    out = []
    for c in candidates[:top_k]:
        if "rerank_score" in c:
            score = c["rerank_score"]
            score_source = "rerank"
        elif "hybrid_score" in c:
            score = c["hybrid_score"]
            score_source = "hybrid"
        else:
            score = c.get("score", 0.0)
            score_source = "dense"
        row = {
            "chunk_id": c["chunk_id"],
            "text": c["text"],
            "page": c["page"],
            "score": score,
            "score_source": score_source,
        }
        row.update(_extras_from_candidate(c))
        out.append(row)
    return out


def run_phase3_retrieval(
    query: str,
    qv: list[float],
    store,
    settings,
    modality_intent: str | None = None,
    visual_store=None,
    visual_query_vector: list[float] | None = None,
) -> list[dict]:
    dense_raw = store.search(qv, top_k=settings.dense_top_k)
    dense_candidates = [
        {
            "chunk_id": r["metadata"]["chunk_id"],
            "text": r["metadata"]["text"],
            "page": r["metadata"]["page"],
            "doc_id": r["metadata"].get("doc_id", ""),
            "score": r["score"],
            "metadata": r["metadata"],
        }
        for r in dense_raw
    ]
    mode = safe_mode_from_flags(settings.enable_hybrid, settings.enable_rerank)
    router_on = getattr(settings, "enable_modality_router", False)
    visual_gate = visual_merge_gate_intent(settings, modality_intent, query)

    def _visual_fuse(cands: list[dict]) -> list[dict]:
        if (
            visual_store is None
            or not visual_query_vector
            or not should_run_visual_merge(settings, visual_gate)
        ):
            return cands
        try:
            vk = int(getattr(settings, "visual_top_k", 20))
            raw = visual_store.search(visual_query_vector, top_k=vk)
        except Exception:
            return cands
        vc = [
            {
                "chunk_id": r["metadata"]["chunk_id"],
                "text": r["metadata"]["text"],
                "page": r["metadata"]["page"],
                "doc_id": r["metadata"].get("doc_id", ""),
                "score": r["score"],
                "metadata": r["metadata"],
            }
            for r in raw
        ]
        if not vc:
            return cands
        lam = float(getattr(settings, "visual_fusion_lambda", 0.65))
        return merge_text_and_visual_candidates(cands, vc, lam)

    if mode == "dense_only":
        merged = _visual_fuse(dense_candidates)
        ri, pref = _rank_intent_and_preference_enabled(
            modality_intent, router_on, visual_gate
        )
        ranked = apply_modality_preference(merged, ri, pref)
        return build_final_context(ranked, top_k=settings.top_k)

    # Sparse path
    sparse_corpus = []
    try:
        if settings.sparse_index_path:
            sparse_corpus = load_sparse_corpus(settings.sparse_index_path)
    except Exception:
        sparse_corpus = []
    if not sparse_corpus:
        sparse_corpus = store.all_metadata()
    sparse = SimpleBM25Retriever(sparse_corpus).search(query, top_k=settings.sparse_top_k)
    fused = fuse_dense_sparse(
        dense_candidates, sparse, alpha=settings.hybrid_alpha, top_n=settings.hybrid_top_n
    )
    fused_merged = _visual_fuse(fused)
    if mode == "hybrid_no_rerank":
        ri, pref = _rank_intent_and_preference_enabled(
            modality_intent, router_on, visual_gate
        )
        ranked = apply_modality_preference(fused_merged, ri, pref)
        return build_final_context(ranked, top_k=settings.top_k)

    # Rerank model (optional, fallback to fused order)
    rerank_model = None
    if settings.enable_rerank:
        try:
            from sentence_transformers import CrossEncoder

            rerank_model = CrossEncoder(settings.rerank_model)
        except Exception:
            rerank_model = None
    rr_cap = min(
        32,
        max(
            len(fused_merged),
            settings.rerank_top_k,
            int(getattr(settings, "visual_top_k", 20)),
        ),
    )
    expand_rerank_pool = visual_gate in ("image", "mixed") or modality_intent in (
        "image",
        "mixed",
    )
    reranked = rerank_candidates(
        query,
        fused_merged,
        top_k=rr_cap,
        model=rerank_model,
        truncate=not expand_rerank_pool,
    )
    ri, pref = _rank_intent_and_preference_enabled(
        modality_intent, router_on, visual_gate
    )
    ranked = apply_modality_preference(reranked, ri, pref)
    return build_final_context(ranked, top_k=settings.top_k)

