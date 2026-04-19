# Phase 3 Hybrid Retrieval + Reranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement full Phase 3 retrieval upgrade by adding BM25 sparse retrieval, weighted hybrid fusion, and cross-encoder reranking while preserving existing app/evaluation compatibility.

**Architecture:** Introduce modular retrieval stages (`bm25`, `hybrid_fusion`, `rerank`) and one orchestrator entrypoint. Keep existing dense-retrieval API as fallback mode. Evaluation reuses Phase 2 dataset/report pipeline for regression comparison.

**Tech Stack:** Python, rank-bm25, sentence-transformers cross-encoder, existing FAISS + BGE pipeline, pytest.

## What this phase uses

| Category | Items |
|----------|--------|
| **Phase 1 outputs** | Dense FAISS index + `faiss.index.meta.json`; same embedder model as indexing |
| **New Python deps** | `rank-bm25`, `scikit-learn` (helpers if used) — see `requirements.txt` |
| **New artifacts** | BM25 corpus JSON (`SPARSE_INDEX_PATH`), produced by `scripts/index_phase3_sparse.py` |
| **Code** | `retrieval/bm25.py`, `hybrid_retriever.py`, `pipeline.py`; `reranking/cross_encoder.py`; `main.run_rag_query` → `run_phase3_retrieval` |
| **Flags** | `ENABLE_HYBRID`, `ENABLE_RERANK`, fusion and pool sizes — all in `configs/settings.py` + `.env.example` |

---

## File Structure (Phase 3)

- Create: `retrieval/bm25.py`
- Create: `retrieval/hybrid_retriever.py`
- Create: `retrieval/pipeline.py`
- Create: `reranking/__init__.py`
- Create: `reranking/cross_encoder.py`
- Create: `scripts/index_phase3_sparse.py`
- Create: `tests/test_bm25.py`
- Create: `tests/test_hybrid_retriever.py`
- Create: `tests/test_reranker.py`
- Create: `tests/test_phase3_pipeline.py`
- Modify: `configs/settings.py`
- Modify: `.env.example`
- Modify: `main.py`
- Modify: `requirements.txt`
- Modify: `README.md`

### Task 1: Config + Dependency Scaffolding

**Files:**
- Modify: `requirements.txt`
- Modify: `configs/settings.py`
- Modify: `.env.example`
- Test: `tests/test_phase3_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_phase3_pipeline.py
from configs.settings import Settings

def test_phase3_settings_defaults_exist():
    s = Settings()
    assert hasattr(s, "hybrid_alpha")
    assert hasattr(s, "enable_hybrid")
    assert hasattr(s, "enable_rerank")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_phase3_pipeline.py::test_phase3_settings_defaults_exist -v`  
Expected: FAIL on missing attributes.

- [ ] **Step 3: Write minimal implementation**

- Add dependencies:
  - `rank-bm25`
  - `scikit-learn` (optional normalization helpers)
- Add settings:
  - `hybrid_alpha`, `dense_top_k`, `sparse_top_k`, `hybrid_top_n`, `rerank_top_k`
  - `enable_hybrid`, `enable_rerank`

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_phase3_pipeline.py::test_phase3_settings_defaults_exist -v`  
Expected: PASS

### Task 2: BM25 Sparse Retriever

**Files:**
- Create: `retrieval/bm25.py`
- Create: `tests/test_bm25.py`
- Test: `tests/test_bm25.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bm25.py
from retrieval.bm25 import SimpleBM25Retriever

def test_bm25_returns_relevant_chunk():
    chunks = [
        {"chunk_id": "c1", "text": "net income increased in fy24", "page": 1, "doc_id": "d"},
        {"chunk_id": "c2", "text": "supply chain delays impacted costs", "page": 2, "doc_id": "d"},
    ]
    r = SimpleBM25Retriever(chunks)
    out = r.search("net income", top_k=1)
    assert out[0]["chunk_id"] == "c1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_bm25.py::test_bm25_returns_relevant_chunk -v`  
Expected: FAIL with missing module/class.

- [ ] **Step 3: Write minimal implementation**

Implement:
- tokenizer (lowercase whitespace split baseline)
- `search(query, top_k)`
- return payload:
  - `chunk_id`, `text`, `page`, `doc_id`, `score`

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_bm25.py -v`  
Expected: PASS

### Task 3: Hybrid Fusion Logic

**Files:**
- Create: `retrieval/hybrid_retriever.py`
- Create: `tests/test_hybrid_retriever.py`
- Test: `tests/test_hybrid_retriever.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hybrid_retriever.py
from retrieval.hybrid_retriever import fuse_dense_sparse

def test_fuse_dense_sparse_merges_scores():
    dense = [{"chunk_id": "c1", "score": 0.9}, {"chunk_id": "c2", "score": 0.3}]
    sparse = [{"chunk_id": "c2", "score": 2.0}, {"chunk_id": "c3", "score": 1.0}]
    out = fuse_dense_sparse(dense, sparse, alpha=0.7, top_n=3)
    ids = [o["chunk_id"] for o in out]
    assert "c1" in ids and "c2" in ids and "c3" in ids
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hybrid_retriever.py::test_fuse_dense_sparse_merges_scores -v`  
Expected: FAIL before implementation.

- [ ] **Step 3: Write minimal implementation**

Implement:
- min-max normalization for dense/sparse score spaces independently
- weighted merge on `chunk_id`
- descending sort by `hybrid_score`

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_hybrid_retriever.py -v`  
Expected: PASS

### Task 4: Cross-Encoder Reranking

**Files:**
- Create: `reranking/cross_encoder.py`
- Create: `tests/test_reranker.py`
- Test: `tests/test_reranker.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reranker.py
from reranking.cross_encoder import rerank_candidates

def test_rerank_candidates_shape():
    candidates = [
        {"chunk_id": "c1", "text": "revenue growth in fy24", "score": 0.4},
        {"chunk_id": "c2", "text": "supply chain issue", "score": 0.6},
    ]
    out = rerank_candidates("revenue growth", candidates, top_k=1, model=None)
    assert len(out) == 1
    assert "chunk_id" in out[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_reranker.py::test_rerank_candidates_shape -v`  
Expected: FAIL with missing module/function.

- [ ] **Step 3: Write minimal implementation**

Implement:
- pluggable model argument
- if model unavailable, fallback to incoming order
- output includes `rerank_score` if model available

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_reranker.py -v`  
Expected: PASS

### Task 5: Retrieval Pipeline Orchestrator Integration

**Files:**
- Create: `retrieval/pipeline.py`
- Modify: `main.py`
- Create: `tests/test_phase3_pipeline.py`
- Test: `tests/test_phase3_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_phase3_pipeline.py
from retrieval.pipeline import build_final_context

def test_build_final_context_schema():
    candidates = [{"chunk_id":"c1","text":"abc","page":1,"hybrid_score":0.8}]
    out = build_final_context(candidates, top_k=1)
    assert out == [{"chunk_id":"c1","text":"abc","page":1,"score":0.8}]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_phase3_pipeline.py::test_build_final_context_schema -v`  
Expected: FAIL before function exists.

- [ ] **Step 3: Write minimal implementation**

Implement pipeline:
- dense retrieval from existing FAISS path
- optional sparse retrieval
- optional hybrid fusion
- optional rerank
- output stable context schema

Wire `main.run_rag_query()` to use Phase 3 pipeline when enabled.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_phase3_pipeline.py -v`  
Expected: PASS

### Task 6: Sparse Index Builder Script

**Files:**
- Create: `scripts/index_phase3_sparse.py`
- Modify: `README.md`
- Test: `tests/test_bm25.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bm25.py
def test_bm25_can_initialize_from_chunk_list():
    from retrieval.bm25 import SimpleBM25Retriever
    chunks = [{"chunk_id":"c1","text":"a b c","page":1,"doc_id":"d"}]
    r = SimpleBM25Retriever(chunks)
    assert r is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_bm25.py::test_bm25_can_initialize_from_chunk_list -v`  
Expected: FAIL if retriever not complete.

- [ ] **Step 3: Write minimal implementation**

Script responsibilities:
- load existing indexed metadata source
- initialize BM25 retriever
- persist sparse artifacts (json/pickle) for reuse

Add README commands for Phase 3 indexing sequence.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_bm25.py -v`  
Expected: PASS

### Task 7: End-to-End Phase 3 Verification

**Files:**
- Modify: `README.md`
- Test: `tests/test_bm25.py`, `tests/test_hybrid_retriever.py`, `tests/test_reranker.py`, `tests/test_phase3_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_phase3_pipeline.py
def test_pipeline_dense_only_fallback():
    from retrieval.pipeline import safe_mode_from_flags
    mode = safe_mode_from_flags(enable_hybrid=False, enable_rerank=False)
    assert mode == "dense_only"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_phase3_pipeline.py::test_pipeline_dense_only_fallback -v`  
Expected: FAIL before fallback helper exists.

- [ ] **Step 3: Write minimal implementation**

Add fallback helpers and docs for:
- dense-only mode
- hybrid-no-rerank mode
- full hybrid+rerank mode

Run Phase 2 evaluation dataset after enabling Phase 3 path and compare reports.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_bm25.py tests/test_hybrid_retriever.py tests/test_reranker.py tests/test_phase3_pipeline.py -v`  
Expected: PASS

## Verification Checklist (Before Marking Phase 3 Complete)

- Phase 3 unit/integration tests pass.
- Sparse index build command works.
- Query pipeline runs in all 3 modes:
  - dense-only
  - hybrid-no-rerank
  - hybrid+rerank
- Phase 2 evaluation dataset can be re-run with Phase 3 path enabled.
- New report generated under `data/eval/reports/` and compared to Phase 2 baseline.

