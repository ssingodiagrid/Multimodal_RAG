# Onboarding — how to understand this project

Use this guide if you are new to the repo and want a **deliberate path** through the markdown docs and the Python code. You do not have to read everything; pick a depth below.

---

## 1. Start here (about 15 minutes)

| Order | What to read | Why |
|------|----------------|-----|
| 1 | [README.md](../README.md) | Setup, how to run the UI, high-level folder layout, phase summaries. |
| 1b | [ARCHITECTURE.md](ARCHITECTURE.md) | **Full stack map**: each task → library or module (e.g. RecursiveCharacterTextSplitter for chunking). |
| 2 | [.env.example] | Every feature flag and path the app understands (skim section headers). |
| 3 | [docs/superpowers/specs/2026-04-16-multimodal-rag-phase1-6-design.md](superpowers/specs/2026-04-16-multimodal-rag-phase1-6-design.md) | **Roadmap only** — read the intro, “Architecture Overview”, and “Module Boundaries” (ingestion / retrieval / generation). Skip long phase checklists on first pass. |

After this, you should know: *PDF in → chunks + indexes → query → retrieve → prompt → Gemini answer*, and that features are gated by **env flags**.

---

## 2. Follow one query through the code (about 30 minutes)

The fastest way to connect behavior to files is **one vertical slice**: *user asks a question in the UI*.

1. Open [ui/app.py](../ui/app.py) and find where it calls **`run_rag_query`** (or loads the index and invokes `main`).
2. Open [main.py](../main.py) and read in order:
   - **`run_rag_query`** — modality router, optional query refinement, embeddings, cache, multi-hop, retrieval, optional ColPali, **`answer_query`**.
   - **`answer_query`** — builds the grounded prompt, calls **`GeminiClient`**.
3. Open [retrieval/pipeline.py](../retrieval/pipeline.py) — **`run_phase3_retrieval`** — dense search, optional BM25 hybrid, rerank, optional visual fusion.
4. Open [generation/prompt_builder.py](../generation/prompt_builder.py) — **`build_grounded_prompt`** — what the model is actually told.

Optional: trace **indexing** from **`index_pdf`** in `main.py` through [ingestion/chunker.py](../ingestion/chunker.py) and the same `main.py` paths that build FAISS.

---

## 3. Use the phase docs the way they were written

Each phase has two companion files:

| Type | Folder | Purpose |
|------|--------|---------|
| **Design (spec)** | `docs/superpowers/specs/` | *What* and *why*: goals, data contracts, config tables, and **## What this phase uses** (libraries, prior phases, key modules, env). |
| **Implementation (plan)** | `docs/superpowers/plans/` | *Where in the repo*: file lists, wiring notes; same **## What this phase uses** (or a pointer to the spec) for stack alignment. |

**Suggested reading order** (spec first, then plan, then open the code files named in the plan):

| Phase | Spec | Plan | Main code areas |
|-------|------|------|-----------------|
| 1 — Text RAG | (covered in multimodal roadmap + README) | [plans/2026-04-16-phase-1-text-rag-implementation.md](superpowers/plans/2026-04-16-phase-1-text-rag-implementation.md) | `ingestion/`, `retrieval/`, `main.py` |
| 2 — Eval | roadmap | [plans/2026-04-17-phase-2-evaluation-implementation.md](superpowers/plans/2026-04-17-phase-2-evaluation-implementation.md) | `scripts/run_phase2_eval.py`, `eval/` |
| 3 — Hybrid + rerank | [specs/2026-04-17-phase-3-hybrid-rerank-design.md](superpowers/specs/2026-04-17-phase-3-hybrid-rerank-design.md) | [plans/2026-04-17-phase-3-hybrid-rerank-implementation.md](superpowers/plans/2026-04-17-phase-3-hybrid-rerank-implementation.md) | `retrieval/pipeline.py`, BM25 / reranker modules |
| 4 — Cache + multi-hop | [specs/2026-04-17-phase-4-cache-multihop-design.md](superpowers/specs/2026-04-17-phase-4-cache-multihop-design.md) | [plans/2026-04-17-phase-4-cache-multihop-implementation.md](superpowers/plans/2026-04-17-phase-4-cache-multihop-implementation.md) | `cache/`, `retrieval/multihop.py`, `main.py` |
| 4 ext — Query refinement | [specs/2026-04-17-phase-4-query-refinement-design.md](superpowers/specs/2026-04-17-phase-4-query-refinement-design.md) | [plans/2026-04-17-phase-4-query-refinement-implementation.md](superpowers/plans/2026-04-17-phase-4-query-refinement-implementation.md) | `generation/query_refinement.py`, `retrieval/dual_query_merge.py` |
| 5 — Tables + images | [specs/2026-04-17-phase-5-tables-images-design.md](superpowers/specs/2026-04-17-phase-5-tables-images-design.md) | [plans/2026-04-17-phase-5-tables-images-implementation.md](superpowers/plans/2026-04-17-phase-5-tables-images-implementation.md) | `ingestion/tables_*`, `ingestion/images_*`, `retrieval/modality_router.py` |
| 6 — Visual + ColPali | [specs/2026-04-17-phase-6-visual-retrieval-design.md](superpowers/specs/2026-04-17-phase-6-visual-retrieval-design.md) | [plans/2026-04-17-phase-6-visual-retrieval-implementation.md](superpowers/plans/2026-04-17-phase-6-visual-retrieval-implementation.md) | `retrieval/visual_*`, `retrieval/colpali_retrieval.py`, `ingestion/colpali_raster.py` |

**Full catalog** of all spec/plan filenames: [docs/superpowers/README.md](superpowers/README.md).

---

## 4. Use tests as executable documentation

Tests are small, deterministic examples of how modules behave.

```bash
cd /path/to/RAG
source .venv/bin/activate
pytest -q --collect-only   # list test modules without running
```

| If you care about… | Start with… |
|--------------------|-------------|
| Chunking / store | `tests/test_chunker.py`, `tests/test_vector_store.py` |
| Retrieval / hybrid | `tests/test_phase3_pipeline.py`, `tests/test_hybrid_retriever.py` |
| Cache / multi-hop | `tests/test_semantic_cache.py`, `tests/test_multihop.py`, `tests/test_phase4_orchestration.py` |
| Query refinement merge | `tests/test_dual_query_merge.py` |
| Prompts | `tests/test_prompt_builder_phase5.py`, `tests/test_retriever.py` |
| ColPali math / wiring | `tests/test_colpali_maxsim.py`, `tests/test_phase6_settings.py` |

Open a test file, read the function name and assertions, then jump to the production module under test.

---

## 5. Configuration mental model

- **`configs/settings.py`** defines the `Settings` object (defaults often from `os.getenv`).
- **`ensure_phase4_fields`**, **`ensure_phase5_fields`**, **`ensure_phase6_fields`**, **`ensure_colpali_fields`** backfill attributes when older objects are loaded (e.g. Streamlit reload).
- **`.env.example`** is the human-readable source of truth for variable names; your local **`.env`** overrides defaults.

When you read a flag in `.env.example`, **grep** the codebase for that string to find where it is used (often only `settings.py` and one branch in `main.py` or `pipeline.py`).

---

## 6. If you get lost

1. Re-center on **`main.run_rag_query`** and **`main.index_pdf`**.
2. Re-read the **implementation plan** (not the spec) for the phase you care about — plans usually list **file paths**.
3. Use **tests** and **`pytest -k partial_name`** to narrow behavior.

---

## 7. Debugging behavior or failures

For a step-by-step approach (bisect optional features with `.env`, where to set breakpoints, Langfuse, common GCP/FAISS/torch issues), use **[DEBUGGING.md](DEBUGGING.md)**.

---

**Summary:** README + multimodal roadmap → one trace through `main.py` → phase **spec → plan → code** → tests and `.env.example`. That sequence uses all markdown and code together without reading files in random order.
