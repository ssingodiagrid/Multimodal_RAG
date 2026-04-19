# Phase 4 Semantic Cache + Multi-Hop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add semantic query cache (embedding + fingerprint) and a two-hop retrieval path with Gemini sub-query generation, integrated into `main.run_rag_query` without breaking Phase 2 eval or Phase 3 retrieval.

**Architecture:** FAISS-backed cache index for query vectors; JSON sidecar for answers/context; fingerprint from corpus index files + embedding model; multi-hop as a thin Python orchestrator calling `run_phase3_retrieval` twice and merging chunks by `chunk_id`.

**Tech Stack:** Existing FAISS + Embedder + `GeminiClient`, pytest, optional tempfile tests.

## What this phase uses

| Category | Items |
|----------|--------|
| **Retrieval** | Unchanged **Phase 3** path: `main._retrieve_context` → `run_phase3_retrieval` |
| **Cache** | `cache/semantic_cache.py` (FAISS + metadata files under `SEMANTIC_CACHE_PATH`); `cache/index_fingerprint.py` |
| **Multi-hop LLM** | `generation/multihop_prompts.py` + `retrieval/multihop.generate_sub_query` → `GeminiClient.answer` |
| **Merge** | `retrieval/multihop.merge_contexts` — dedupe by `chunk_id`, cap `multi_hop_merged_top_k` |
| **Wire-in** | `main.run_rag_query` (cache lookup/store, hop branches); `configs/settings.py` + `ensure_phase4_fields` |
| **Tests** | `tests/test_semantic_cache.py`, `tests/test_multihop.py`, `tests/test_index_fingerprint.py`, `tests/test_phase4_orchestration.py` |

Extension (**query refinement**): additional modules listed in [`2026-04-17-phase-4-query-refinement-implementation.md`](./2026-04-17-phase-4-query-refinement-implementation.md).

---

## File structure (Phase 4)

- Create: `cache/__init__.py`
- Create: `cache/index_fingerprint.py`
- Create: `cache/semantic_cache.py`
- Create: `retrieval/multihop.py`
- Create: `generation/multihop_prompts.py`
- Create: `tests/test_index_fingerprint.py`
- Create: `tests/test_semantic_cache.py`
- Create: `tests/test_multihop.py`
- Create: `tests/test_phase4_orchestration.py` (light integration against `run_rag_query` with mocks)
- Modify: `configs/settings.py`
- Modify: `.env.example`
- Modify: `main.py`
- Modify: `README.md`
- Optional modify: `ui/app.py` (sidebar toggles or debug expander for `cache_hit` / `sub_query`)

---

### Task 1: Settings + env scaffolding

**Files:**

- Modify: `configs/settings.py`
- Modify: `.env.example`
- Create: `tests/test_phase4_settings.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_phase4_settings.py
from configs.settings import Settings

def test_phase4_cache_settings_exist():
    s = Settings()
    assert hasattr(s, "enable_semantic_cache")
    assert hasattr(s, "semantic_cache_threshold")
    assert hasattr(s, "semantic_cache_max_entries")
    assert hasattr(s, "semantic_cache_path")

def test_phase4_multihop_settings_exist():
    s = Settings()
    assert hasattr(s, "enable_multi_hop")
    assert hasattr(s, "multi_hop_mode")
    assert hasattr(s, "multi_hop_merged_top_k")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_phase4_settings.py -v`  
Expected: FAIL on missing attributes.

- [ ] **Step 3: Write minimal implementation**

Add to `Settings`:

- `enable_semantic_cache: bool` — `ENABLE_SEMANTIC_CACHE`, default `false`
- `semantic_cache_threshold: float` — `SEMANTIC_CACHE_THRESHOLD`, default `0.92`
- `semantic_cache_max_entries: int` — `SEMANTIC_CACHE_MAX_ENTRIES`, default `500`
- `semantic_cache_path: str` — `SEMANTIC_CACHE_PATH`, default `data/cache/semantic_cache`
- `enable_multi_hop: bool` — `ENABLE_MULTI_HOP`, default `false`
- `multi_hop_mode: str` — `MULTI_HOP_MODE`, default `heuristic` (validate allowed: `off`, `heuristic`, `always`)
- `multi_hop_merged_top_k: int` — `MULTI_HOP_MERGED_TOP_K`, default mirror `TOP_K`

Document each in `.env.example`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_phase4_settings.py -v`  
Expected: PASS

---

### Task 2: Index fingerprint

**Files:**

- Create: `cache/index_fingerprint.py`
- Create: `tests/test_index_fingerprint.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_index_fingerprint.py
import tempfile
from pathlib import Path

from cache.index_fingerprint import compute_index_fingerprint


def test_fingerprint_changes_when_file_changes():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "fake.faiss"
        p.write_bytes(b"a")
        fp1 = compute_index_fingerprint(str(p), sparse_path=None, embedding_model="m1")
        p.write_bytes(b"ab")
        fp2 = compute_index_fingerprint(str(p), sparse_path=None, embedding_model="m1")
        assert fp1 != fp2
```

- [ ] **Step 2: Run test — expect FAIL** (module missing).

- [ ] **Step 3: Implement `compute_index_fingerprint(faiss_path, sparse_path, embedding_model) -> str`**

- If `faiss_path` file missing, return hash of string `"missing:" + faiss_path + embedding_model` (deterministic miss).
- Else incorporate: resolved path, file size, mtime_ns (or mtime float), `embedding_model`, and if `sparse_path` provided and file exists same for sparse.
- Return `hashlib.sha256(...).hexdigest()` of a canonical joined string.

- [ ] **Step 4: Run test — expect PASS**

---

### Task 3: Semantic cache core

**Files:**

- Create: `cache/semantic_cache.py`
- Create: `tests/test_semantic_cache.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_semantic_cache.py
import tempfile
from pathlib import Path

from cache.semantic_cache import SemanticCache


def test_cache_miss_empty():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d) / "sc"
        c = SemanticCache(dim=4, base_path=str(base), threshold=0.99, max_entries=10)
        hit = c.lookup([1.0, 0.0, 0.0, 0.0], fingerprint="fp1")
        assert hit is None


def test_cache_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d) / "sc"
        c = SemanticCache(dim=3, base_path=str(base), threshold=0.95, max_entries=10)
        qv = [0.6, 0.8, 0.0]
        c.store(qv, fingerprint="fp1", query_text="q", answer="a", context=[{"chunk_id": "1", "text": "t", "page": 1, "score": 1.0, "score_source": "dense"}])
        hit = c.lookup(qv, fingerprint="fp1")
        assert hit is not None
        assert hit.answer == "a"
        assert hit.context[0]["chunk_id"] == "1"
```

Normalize vectors in cache same way as corpus (L2 norm) before add/search if your `FaissVectorStore` expects unit vectors — match `Embedder` / `FaissVectorStore` convention.

- [ ] **Step 2: Implement `SemanticCache`**

- Reuse `FaissVectorStore` from `retrieval/vector_store.py` or duplicate minimal IndexFlatIP + meta list **only if** coupling is awkward; prefer reuse.
- Sidecar: JSON list or JSONL of records keyed by meta index position; on load rebuild FAISS + meta.
- `lookup`: search top-1; if `score < threshold` return `None`; if fingerprint in meta != requested fingerprint return `None`.
- `store`: append; if `len > max_entries` remove oldest (FIFO: drop index 0 and rebuild, or keep deque of ids — document simplest rebuild).

- [ ] **Step 3: Tests PASS**

---

### Task 4: Multi-hop helpers

**Files:**

- Create: `generation/multihop_prompts.py`
- Create: `retrieval/multihop.py`
- Create: `tests/test_multihop.py`

- [ ] **Step 1: Tests first**

```python
# tests/test_multihop.py
from retrieval.multihop import merge_contexts, should_multihop


def test_merge_dedupes_chunk_id():
    a = [{"chunk_id": "1", "text": "x", "page": 1, "score": 0.9, "score_source": "dense"}]
    b = [{"chunk_id": "1", "text": "x", "page": 1, "score": 0.5, "score_source": "dense"}, {"chunk_id": "2", "text": "y", "page": 2, "score": 0.8, "score_source": "dense"}]
    m = merge_contexts(a, b, max_chunks=3)
    assert len(m) == 2
    assert m[0]["chunk_id"] == "1"


def test_should_multihop_heuristic():
    class S:
        enable_multi_hop = True
        multi_hop_mode = "heuristic"

    assert should_multihop("What is revenue?", S()) is False
    assert should_multihop("Compare revenue and costs for FY24", S()) is True
```

- [ ] **Step 2: Implement `merge_contexts`**, **`should_multihop`**, **`parse_sub_query_json(text) -> str | None`**

- `generation/multihop_prompts.py`: `build_sub_query_prompt(user_query: str, context_chunks: list[dict]) -> str`

- [ ] **Step 3: Add `generate_sub_query(..., GeminiClient)`** in `retrieval/multihop.py` calling `llm.answer(prompt)` and `parse_sub_query_json`.

- [ ] **Step 4: Tests PASS** (mock LLM not required if `parse_sub_query_json` tested separately with raw strings).

---

### Task 5: Wire `main.run_rag_query`

**Files:**

- Modify: `main.py`
- Create: `tests/test_phase4_orchestration.py`

- [ ] **Step 1: Refactor minimally**

- Extract private helper `_retrieve_and_answer(...)` if needed to avoid duplication between single-hop and multi-hop.
- Order of operations:
  1. `fingerprint = compute_index_fingerprint(...)` using paths from `Settings` (sparse path if hybrid).
  2. If `enable_semantic_cache`: `cache.lookup(qv, fingerprint)`; on hit return `(hit.answer, hit.context)` — optionally annotate context with `{"cache_hit": True}` in trace only to avoid polluting schema (or add as optional key — spec allows trace).

- [ ] **Step 2: Multi-hop branch**

- If cache miss and `enable_multi_hop` and `should_multihop`:
  - `ctx1 = run_phase3_retrieval(query, qv, store, settings)`
  - `sq = generate_sub_query(...)`; if `sq`: `ctx2 = run_phase3_retrieval(sq, embedder.embed_query(sq), store, settings)` else `ctx2 = []`
  - `merged = merge_contexts(ctx1, ctx2, max_chunks=settings.multi_hop_merged_top_k)`
  - `answer = answer_query(query, merged, settings)`
- Else existing single path.

- [ ] **Step 3: On cache miss and cache enabled**, call `cache.store(...)` after successful answer (use same `qv`, fingerprint, query, answer, context).

- [ ] **Step 4: Integration test** with mocks: patch `run_phase3_retrieval` to return fixed lists; patch `answer_query`; assert multi-hop calls retrieval twice when `should_multihop` true.

- [ ] **Step 5: `pytest` full suite** — no regressions.

---

### Task 6: Documentation + UI (optional debug)

**Files:**

- Modify: `README.md`
- Optional: `ui/app.py`

- [ ] **Step 1: README section "Phase 4"** — env vars, cache invalidation behavior, multi-hop modes, note that eval should use `ENABLE_SEMANTIC_CACHE=false` for metric comparability unless measuring cache.

- [ ] **Step 2 (optional):** Streamlit sidebar checkboxes or text showing last `cache_hit` / `sub_query` from `st.session_state` if returned from `run_rag_query` — **if** you extend return type, do it in a backward-compatible way (e.g. internal session dict set in `main` only when streamlit imported — prefer **avoid**; use Langfuse trace only for v1).

---

## Verification checklist (before marking Phase 4 complete)

- [ ] All new tests pass; full `pytest` green.
- [ ] With defaults (`ENABLE_SEMANTIC_CACHE=false`, `ENABLE_MULTI_HOP=false`), behavior matches Phase 3-only path.
- [ ] Manual: same question twice with cache on shows faster second response (optional timing print).
- [ ] Manual: comparative question with multi-hop on retrieves broader context (inspect Streamlit JSON context).

---

## Extension (2026-04-17): Query refinement + dual retrieval

Documented and implemented separately:

- **Spec:** [`docs/superpowers/specs/2026-04-17-phase-4-query-refinement-design.md`](../specs/2026-04-17-phase-4-query-refinement-design.md)
- **Plan / code map:** [`docs/superpowers/plans/2026-04-17-phase-4-query-refinement-implementation.md`](./2026-04-17-phase-4-query-refinement-implementation.md)

Key modules: `generation/query_refinement.py`, `retrieval/dual_query_merge.py`, `main.run_rag_query`, `generation/prompt_builder.build_grounded_prompt`, `configs/settings.enable_query_refinement`, `.env.example` (`ENABLE_QUERY_REFINEMENT`).
