# Phase 4 extension — Query refinement + dual retrieval (implementation)

**Design reference:** [`docs/superpowers/specs/2026-04-17-phase-4-query-refinement-design.md`](../specs/2026-04-17-phase-4-query-refinement-design.md)

**Status:** Implemented in-repo (2026-04-17).

## What this phase uses

| Category | Items |
|----------|--------|
| **Depends on** | Phase 3 retrieval pipeline, Phase 4 `run_rag_query` control flow, Vertex **Gemini** (same `Settings.gcp_*` / `gemini_model` as the rest of the app) |
| **New code** | `generation/query_refinement.py`, `retrieval/dual_query_merge.py`; branches + `dual_cap` logic in `main.run_rag_query` |
| **Prompt / answer** | `build_grounded_prompt(..., refined_query=)`, `answer_query(..., refined_query_text=)` |
| **Tests** | `tests/test_dual_query_merge.py`, `tests/test_prompt_builder_phase5.py` (`test_prompt_refinement_hint`) |
| **Env** | `ENABLE_QUERY_REFINEMENT` in `.env.example` |

---

## Code map (authoritative)

| Responsibility | File(s) |
|----------------|---------|
| Settings flag + hot-reload backfill | `configs/settings.py` — `enable_query_refinement`, `ensure_phase4_fields()` |
| LLM rewrite (Vertex `GeminiClient`) | `generation/query_refinement.py` — `refine_search_query()` |
| Merge raw + refined retrieval lists | `retrieval/dual_query_merge.py` — `merge_dual_retrieval_contexts()` |
| Orchestration: refine → dual embed → dual retrieve → multihop / ColPali / cache rules | `main.py` — `run_rag_query()`, helper `_retrieve_effective()` inside that function |
| Thin wrapper around Phase 3 | `main.py` — `_retrieve_context()` → `retrieval/pipeline.py` — `run_phase3_retrieval()` |
| Multi-hop sub-query (unchanged contract) | `retrieval/multihop.py`, `generation/multihop_prompts.py` |
| Final prompt: original question + optional rewrite line | `generation/prompt_builder.py` — `build_grounded_prompt(..., refined_query=)` |
| Final answer entry | `main.py` — `answer_query(..., refined_query_text=)` |
| Env template | `.env.example` — `ENABLE_QUERY_REFINEMENT` |
| Unit tests | `tests/test_dual_query_merge.py`, `tests/test_prompt_builder_phase5.py` (`test_prompt_refinement_hint`) |

---

## Execution order in `run_rag_query` (relevant fragment)

1. `effective_query`, modality router → `modality_intent` / `mi_retrieval`.
2. If `enable_query_refinement` and no `query_image_bytes`: `refined_q = refine_search_query(...)`.
3. `qv_raw = embedder.embed_query(effective_query)`; if `refined_q`: `qv_refined = embedder.embed_query(refined_q)` (on failure, drop refined path).
4. Visual store / `visual_qv` (still driven by **raw** `effective_query` or uploaded image).
5. Semantic cache: **disabled** if `enable_query_refinement` (alongside existing ColPali / image rules).
6. **Single-hop:** `merge_dual_retrieval_contexts(ctx_raw, ctx_refined, top_k=TOP_K)` or raw-only.
7. **Multi-hop:** dual-merge hop 1 with `dual_cap = min(HYBRID_TOP_N, max(TOP_K*2, MULTI_HOP_MERGED_TOP_K))`; `generate_sub_query(effective_query, ctx1[:TOP_K], ...)`; hop 2 single retrieve; `merge_contexts` as before.
8. Optional ColPali merge still uses **raw** `effective_query` for page search (see `main.py`).
9. `answer_query(..., refined_query_text=rq if dual_ok else None)`.

---

## Verification

```bash
pytest tests/test_dual_query_merge.py tests/test_prompt_builder_phase5.py tests/test_phase4_orchestration.py -q
```

Full suite: `pytest -q`

---

## Checklist (post-merge)

- [x] `ENABLE_QUERY_REFINEMENT` in `.env.example`
- [x] Default `false` to avoid surprise cost/latency
- [x] Cache bypass when refinement on
- [x] Skip refinement for query-image path
- [x] Document merge policy (`retrieval_source`, max score)
