# Phase 5 Tables + Images Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add table extraction + normalization and image extract + Gemini captioning to the PDF indexing path; merge new chunks into the existing FAISS/BM25 stack; add an optional modality router and prompt formatting; **extend retrieval output and Streamlit so retrieved tables and images render in the UI** (dataframe + thumbnails), not only as JSON text. Phase 5 remains caption/table-text retrieval only (no CLIP).

**Architecture:** PyMuPDF for images and base PDF I/O; Camelot (optional) + pdfplumber fallback for tables; extend `index_pdf` to build a unified chunk list; extend Phase 3 pipeline with optional modality bias; extend **`build_final_context`** / dense fallback to **pass through** `modality`, `asset_path`, `table_json`; extend prompt builder for tagged context lines; **Streamlit** renders each retrieved chunk with `st.image` / `st.dataframe` when metadata allows.

**Tech Stack:** Existing Vertex Gemini, sentence-transformers, FAISS, BM25; new optional deps `camelot-py`, `pdfplumber`, `Pillow` (resize).

## What this phase uses

| Category | Items |
|----------|--------|
| **Indexing** | `main.index_pdf` extended — `ingestion/tables_extract.py`, `tables_normalize.py`, `images_extract.py`, `generation/image_caption.py`, optional `page_render_extract.py` |
| **Retrieval** | `retrieval/modality_router.py`, `modality_rank.py`, hooks in **`retrieval/pipeline.py`** (`build_final_context` passes `modality`, `asset_path`, `table_json`, …) |
| **Generation** | `generation/prompt_builder.py`; multimodal **Gemini** for captions in `image_caption.py` |
| **UI** | `ui/context_evidence.py`, `ui/app.py` |
| **Config** | `configs/settings.py` + `ensure_phase5_fields`; flags in `.env.example` (table/image/router/assets/DPI/strategy) |
| **Tests** | `tests/test_tables_normalize.py`, `tests/test_modality_router.py`, `tests/test_prompt_builder_phase5.py`, `tests/test_build_final_context_phase5.py`, merge/mocks as listed in tasks below |

---

## File structure

- Create: `ingestion/tables_extract.py`
- Create: `ingestion/tables_normalize.py`
- Create: `ingestion/images_extract.py`
- Create: `generation/image_caption.py`
- Create: `retrieval/modality_router.py`
- Create: `tests/test_tables_normalize.py`
- Create: `tests/test_modality_router.py`
- Create: `tests/test_prompt_builder_phase5.py`
- Create: `tests/test_phase5_merge_chunks.py` (mocked extractors)
- Modify: `configs/settings.py`
- Modify: `.env.example`
- Modify: `main.py` (`index_pdf`, optionally `run_rag_query` for router)
- Modify: `retrieval/pipeline.py` (modality bias hook)
- Modify: `generation/prompt_builder.py`
- Modify: `generation/llm_pipeline.py` or only `image_caption.py` (multimodal `generate_content`)
- Modify: `requirements.txt` (optional extras commented or separate `requirements-phase5.txt` — prefer optional deps in README)
- Modify: `README.md`
- Create: `ui/context_evidence.py` (recommended) — `render_retrieved_chunk(chunk: dict, settings)` for Streamlit
- Modify: `ui/app.py` — call renderer inside “Retrieved context”; optional nested “Raw metadata” `st.json`
- Modify: `retrieval/pipeline.py` — **`build_final_context`**: copy modality fields from each candidate into output dicts
- Modify: `retrieval/retriever.py` — if dense fallback should support UI later, pass through metadata keys from `item["metadata"]` where applicable (or document text-only fallback)

---

### Task 1: Settings + dependencies documentation

**Files:** `configs/settings.py`, `.env.example`, `README.md`, `requirements.txt` (optional line)

- [ ] **Step 1: Failing test** — `tests/test_phase5_settings.py`:

```python
from configs.settings import Settings

def test_phase5_flags_exist():
    s = Settings()
    assert hasattr(s, "enable_table_extraction")
    assert hasattr(s, "enable_image_captions")
    assert hasattr(s, "enable_modality_router")
```

- [ ] **Step 2:** `pytest tests/test_phase5_settings.py` → FAIL

- [ ] **Step 3:** Add fields: `enable_table_extraction`, `enable_image_captions`, `enable_modality_router`, `router_use_llm`, `table_max_rows_per_chunk`, `image_caption_max_side`, `assets_dir`, env keys as in design spec.

- [ ] **Step 4:** Document Camelot/Ghostscript/Java and optional `pdfplumber` in README; add pip extras note.

---

### Task 2: Table normalization (pure logic, TDD)

**Files:** `ingestion/tables_normalize.py`, `tests/test_tables_normalize.py`

- [ ] **Step 1:** Test: given a small `headers` + `rows` matrix, `table_to_chunks(...)` returns list of dicts with `text`, `modality="table"`, `page`, `table_id`, `chunk_id` prefix.

- [ ] **Step 2:** Implement `table_to_chunks(doc_id, page, table_id, headers, rows, max_rows_per_chunk) -> list[dict]`

- [ ] **Step 3:** `pytest tests/test_tables_normalize.py` → PASS

---

### Task 3: Table extraction adapter

**Files:** `ingestion/tables_extract.py`, `tests/test_tables_extract.py` (mock camelot/pdfplumber)

- [ ] **Step 1:** Implement `extract_tables_from_pdf(pdf_path: str) -> list[dict]` each item `{page, table_index, headers, rows}` — try `camelot.read_pdf`, on `ImportError` or empty try `pdfplumber` if installed, else return `[]`.

- [ ] **Step 2:** Unit test with monkeypatched extractors returning one synthetic table.

---

### Task 4: Image extract + caption

**Files:** `ingestion/images_extract.py`, `generation/image_caption.py`, `tests/test_images_extract.py` (tmpdir)

- [ ] **Step 1:** `extract_images_from_pdf(pdf_path, assets_dir, doc_id, max_side) -> list[dict]` using PyMuPDF: save PNG/JPEG, return `{page, image_index, path, mime}`.

- [ ] **Step 2:** `caption_image_with_gemini(path, mime, settings) -> str` using `GenerativeModel` + `Part.from_data` (Vertex SDK pattern); test with mock `generate_content`.

- [ ] **Step 3:** `image_to_chunk(doc_id, page, idx, caption, asset_path) -> dict` with `modality="image"`.

---

### Task 5: Merge into `index_pdf`

**Files:** `main.py`, `tests/test_phase5_merge_chunks.py`

- [ ] **Step 1:** Refactor `index_pdf` to build `chunks = chunk_pages(...)` then `if settings.enable_table_extraction: chunks += tables_to_chunks(...)` and same for images when `enable_image_captions`.

- [ ] **Step 2:** Ensure `chunk_id` uniqueness across modalities.

- [ ] **Step 3:** Integration test: patch extractors to return empty; assert chunk count equals text-only baseline. Patch to return one table chunk; assert +1.

---

### Task 6: Modality router

**Files:** `retrieval/modality_router.py`, `tests/test_modality_router.py`

- [ ] **Step 1:** `ModalityIntent` enum or string literal union: `text|table|image|mixed`.

- [ ] **Step 2:** `route_query_heuristic(query) -> ModalityIntent` — table vs image keyword lists.

- [ ] **Step 3:** Optional `route_query_llm` behind `router_use_llm` (mock in test).

---

### Task 7: Pipeline bias + prompt builder + **context schema for UI**

**Files:** `retrieval/pipeline.py`, `generation/prompt_builder.py`, `retrieval/retriever.py` (optional), tests

- [ ] **Step 1:** `build_grounded_prompt`: if chunk has `modality=="table"`, prefix line with `[table p{n}]`; if `image`, `[figure p{n}]` + optional path.

- [ ] **Step 2:** In `run_phase3_retrieval` or `build_final_context` precursor: if `enable_modality_router` and intent != `text`, boost or filter fused candidates with matching `modality` in metadata (document exact rule: e.g. sort key `(modality_match, hybrid_score)`).

- [ ] **Step 3:** **`build_final_context`:** extend each output dict to include optional keys preserved from the candidate: at minimum `modality`, `asset_path`, `table_json`, `table_id`, `image_id`, `doc_id` (when present on the incoming candidate / metadata). Falsy keys may be omitted for JSON size.

- [ ] **Step 4:** Tests for prompt lines; test bias function in isolation; **test that `build_final_context` preserves `table_json` and `asset_path`** on a synthetic candidate.

---

### Task 8: Wire `run_rag_query`

**Files:** `main.py`

- [ ] **Step 1:** Before retrieval, `intent = route_query(...)` if router enabled; pass `intent` into `run_phase3_retrieval` (add optional parameter).

- [ ] **Step 2:** Ensure dense fallback path ignores bias safely.

---

### Task 9: UI — **render retrieved tables & images** + cleanup on reindex

**Files:** `ui/context_evidence.py`, `ui/app.py`

- [ ] **Step 1:** When resetting index, delete `data/parsed/assets/{doc_id}` or full `assets_dir` children for prior doc (match Phase 4 cache wipe pattern — wipe assets subtree for indexed doc or entire `ASSETS_DIR` for simplicity).

- [ ] **Step 2:** Implement **`render_retrieved_chunk(chunk, settings)`**:
  - Resolve **`asset_path`**: try `Path(asset_path)` if absolute; else `Path.cwd() / asset_path` or repo root + `settings.assets_dir` (match how indexing writes paths — document one canonical rule in code comments).
  - **`modality == "image"`** (or `asset_path` set): **`st.image`** + caption from `text` + page; if file missing, `st.warning` + show caption.
  - **`modality == "table"`** and **`table_json`** with `headers`/`rows`: **`st.dataframe`** (pandas) or **`st.table`**; else show **`text`** in `st.markdown` / code block.
  - Default: show text preview + metadata line (`chunk_id`, `page`, `score`, `score_source`).

- [ ] **Step 3:** In `_render_chat_history`, inside “Retrieved context”, **loop** `for i, item in enumerate(context):` → subheader or divider `Evidence {i+1}` → **`render_retrieved_chunk(item, settings)`** → optional collapsed **`st.json(item)`** (“Raw metadata”).

- [ ] **Step 4:** With Phase 5 off, context items are text-only; renderer must **not error** (only text branch used).

- [ ] **Step 5 (tests):** `tests/test_build_final_context_phase5_fields.py` or extend existing pipeline tests; optional `tests/test_context_evidence.py` with mocked `streamlit` or pure logic extracting “should use dataframe vs image” from chunk dicts.

---

## Verification checklist

- [ ] `pytest` full suite green; Phase 5 tests skip network by default.
- [ ] With all Phase 5 flags `false`, indexing and query unchanged vs Phase 4; UI still renders text chunks without layout errors.
- [ ] Manual: PDF with table → table chunks in `faiss.index.meta.json`; query retrieves them; **UI shows a table grid** when `table_json` is present.
- [ ] Manual: PDF with figure → caption chunks; **UI shows image thumbnail** when `asset_path` resolves.
- [ ] Manual: missing image file → graceful fallback (caption + warning), no traceback.

---

**Next phase:** [Phase 6 visual retrieval — implementation plan](../plans/2026-04-17-phase-6-visual-retrieval-implementation.md) · [Phase 6 design spec](../specs/2026-04-17-phase-6-visual-retrieval-design.md)
