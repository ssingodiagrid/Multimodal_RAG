# Phase 6 Visual Retrieval — Implementation Plan

> **For agentic workers:** Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to execute tasks in order. Steps use `- [ ]` checkboxes.

**Goal:** Introduce an **optional CLIP-class visual embedding index** aligned to Phase 5 **`chunk_id`** for `modality=image` rows, **fuse** visual retrieval scores with the existing Phase 3 dense + hybrid + rerank pipeline, and preserve **full Phase 5 fallback** when disabled or broken.

**Design reference:** [`docs/superpowers/specs/2026-04-17-phase-6-visual-retrieval-design.md`](../specs/2026-04-17-phase-6-visual-retrieval-design.md)

## What this phase uses

Authoritative breakdown (libraries, artifacts, env): **["What this phase uses" in the Phase 6 design spec](../specs/2026-04-17-phase-6-visual-retrieval-design.md#what-this-phase-uses)**.

**In short:** **Phase 5** assets and `chunk_id`; **Phase 3** `run_phase3_retrieval` plus **CLIP** second index + **`retrieval/visual_*`** fusion; **ColPali** optional stack (**`torch`**, **`transformers`**, **`ingestion/colpali_raster.py`**, **`retrieval/colpali_retrieval.py`**); **Vertex Gemini** for text and multi-image answers; **`cache/index_fingerprint.py`** extended for visual (and index paths relevant to cache invalidation).

---

**Status (2026-04-17):** Phase 6.0 **CLIP visual index + fusion** is implemented (settings, visual FAISS at index time, CLIP query embedding + fusion in `retrieval/pipeline.run_phase3_retrieval`, fingerprint extension, UI index reset).

**ColPali (late interaction)** — also implemented: optional full-page index at PDF ingest, MaxSim retrieval, merged `colpali_page` chunks in context, Gemini multi-image answer path. **Code map:**

| Area | File(s) |
|------|---------|
| Settings + `ensure_colpali_fields` | `configs/settings.py` |
| Page PNG rasters for indexing | `ingestion/colpali_raster.py` |
| Build index + MaxSim search | `retrieval/colpali_retrieval.py` |
| Torch device helper (MPS/CUDA) | `retrieval/torch_device.py` |
| Orchestration (search + merge + answer) | `main.py` (`index_pdf`, `run_rag_query`, `answer_query`, `_merge_colpali_into_context`) |
| Prompt hint for ColPali pages | `generation/prompt_builder.py` |
| Multimodal LLM | `generation/llm_pipeline.py` — `answer_with_images` |
| Rebuild script | `scripts/rebuild_colpali.py` |
| UI evidence row | `ui/context_evidence.py` |
| Tests | `tests/test_colpali_maxsim.py` (if present), `tests/test_phase6_settings.py`, `tests/test_prompt_builder_phase5.py` |

**Non-goal / follow-on:** Qwen2-VL-native answer path; table-as-image-only encoding.

---

## File structure (target)

- Create: `retrieval/visual_embedder.py` — lazy load CLIP; `embed_image_paths`, `embed_query_text`
- Create: `retrieval/visual_fusion.py` — merge candidates by `chunk_id`, RRF or weighted fusion
- Create: `tests/test_visual_fusion.py` — pure functions, no GPU
- Create: `tests/test_phase6_settings.py` — env flags exist
- Modify: `configs/settings.py` — Phase 6 fields + `ensure_phase6_fields`
- Modify: `main.py` — optional second pass after `store.save` **or** dedicated `index_visual_faiss(...)` invoked from `index_pdf` when flags on
- Modify: `retrieval/pipeline.py` — hook after dense candidates, before or after hybrid depending on chosen design (document in code)
- Modify: `cache/index_fingerprint.py` — include visual index path + model id when present
- Modify: `.env.example` — Phase 6 vars commented
- Modify: `README.md` — short “Phase 6 optional” subsection (install extras, rebuild both indexes)

---

## Task 1: Settings + fingerprint

**Files:** `configs/settings.py`, `tests/test_phase6_settings.py`, `cache/index_fingerprint.py`, `.env.example`

- [ ] **Step 1:** Failing test `test_phase6_flags_exist` asserting `hasattr(Settings, "enable_visual_retrieval")` (or agreed name), `visual_faiss_index_path`, `visual_embedding_model`, `visual_top_k`, `visual_fusion_lambda`.

- [ ] **Step 2:** Implement dataclass fields + `ensure_phase6_fields` mirroring Phase 5 pattern.

- [ ] **Step 3:** Extend `compute_index_fingerprint` to optionally hash visual index path + visual model string so semantic cache keys stay correct.

- [ ] **Step 4:** `pytest tests/test_phase6_settings.py` → PASS.

---

## Task 2: Visual embedder (CLIP-class)

**Files:** `retrieval/visual_embedder.py`, `requirements-phase6.txt` (optional) or commented lines in `requirements.txt`

- [ ] **Step 1:** Define a narrow protocol: `VisualEmbedder.embed_images(paths: list[Path]) -> list[list[float]]`, `embed_query(text: str) -> list[float]`, **L2-normalize** outputs to match `IndexFlatIP` usage.

- [ ] **Step 2:** Implement with **lazy import** of `torch` + `open_clip` or `sentence_transformers` CLIP wrapper — import error → raise clear `ImportError` message (“install phase6 extras”).

- [ ] **Step 3:** Unit test with **mocked** forward (patch model) returning fixed vectors for two images + one query; assert shapes and normalization.

- [ ] **Step 4:** Document default model id and VRAM/CPU expectations in module docstring.

---

## Task 3: Build visual FAISS at index time

**Files:** `main.py` (or `ingestion/visual_index.py` if cleaner)

- [ ] **Step 1:** After Phase 5 chunk list is finalized, **filter** `chunks` where `modality == "image"` and `asset_path` present; resolve path relative to repo root; drop missing files with warning.

- [ ] **Step 2:** Batch embed images; build `FaissVectorStore(clip_dim)`; metadata = subset of `_meta_from_chunk` required for retrieval + UI.

- [ ] **Step 3:** Save to `VISUAL_FAISS_INDEX_PATH` sibling files (`.faiss` + `.meta.json`).

- [ ] **Step 4:** If no image chunks or all paths missing, **skip** writing empty index and log info.

- [ ] **Step 5:** Integration test with temp dir: 1–2 tiny PNGs + synthetic metadata list → save/load visual index round-trip (optional; may be slow — gate with `@pytest.mark.slow`).

---

## Task 4: Query-time visual search + fusion

**Files:** `retrieval/visual_fusion.py`, `retrieval/pipeline.py`, `main.py` (`run_rag_query` / `_retrieve_context`)

- [ ] **Step 1:** Implement `visual_search(query, visual_store, embedder, top_k) -> list[dict]` same shape as dense candidates (`chunk_id`, `text`, `page`, `score`, `metadata`).

- [ ] **Step 2:** Implement fusion: **input** dense list + visual list; **output** ordered list capped at `dense_top_k` or `hybrid_top_n` per product decision; document tie-break (prefer higher fused score, then dense).

- [ ] **Step 3:** Wire into `run_phase3_retrieval` **only** when `settings.enable_visual_retrieval` and (router intent in `image`, `mixed` **or** `VISUAL_ALWAYS=false` default requires intent — match design).

- [ ] **Step 4:** Ensure `build_final_context` still receives `modality` / `asset_path` / `pdf_caption` unchanged.

- [ ] **Step 5:** Unit tests for fusion edge cases: empty visual list; no overlap chunk_ids; duplicate chunk_ids.

---

## Task 5: Streamlit + operational UX

**Files:** `ui/app.py` (optional), `README.md`

- [ ] **Step 1:** Optional expander “Visual retrieval: on/off” reflecting env (read-only) for debugging.

- [ ] **Step 2:** README: how to rebuild **text + visual** indexes; disk size note; link to spec.

---

## Task 6: Evaluation hook (lightweight)

**Files:** `scripts/` or extend Phase 2 harness

- [ ] **Step 1:** Document a **manual** eval protocol: N image questions, log whether top-5 contains correct `chunk_id` with vs without visual retrieval.

- [ ] **Step 2 (optional):** JSONL fixture + script comparing recall@k — defer if timeboxed.

---

## Verification checklist

- [ ] `pytest` full suite green; Phase 6 unit tests **do not require** network or GPU by default.
- [ ] `ENABLE_VISUAL_RETRIEVAL=false` → **no** torch import on query path (verify with lazy load).
- [ ] Manual: IFC or sample PDF with Phase 5 image chunks → build visual index → query “Figure 8 …” → image chunk appears in top context more reliably than caption-only baseline.
- [ ] Missing visual index file → pipeline logs warning and behaves as Phase 5.

---

## Phase 6.1+ backlog (do not block 6.0 merge)

- [ ] ColPali / late-interaction retriever; patch grid; MaxSim.
- [ ] Vertex multimodal **answer** generation using retrieved `asset_path` bytes.
- [ ] Table crop rendering + CLIP indexing.

---

**Prerequisite:** Phase 5 image chunks and assets exist (`ENABLE_IMAGE_CAPTIONS`, optional `ENABLE_IMAGE_PAGE_RENDERS`). Visual index is **meaningless** without image rows.
