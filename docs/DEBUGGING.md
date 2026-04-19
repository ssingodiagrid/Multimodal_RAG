# Debugging this codebase — practical approach

Use this when something fails or behaves wrong (indexing, retrieval, empty context, Gemini errors, crashes). The goal is to **narrow the layer** quickly, then **read or instrument** only that code path.

---

## 1. Decide which path you are on

| Symptom | Start here |
|---------|------------|
| Failure while **uploading / indexing** a PDF | `main.index_pdf` → follow calls into `ingestion/*`, then FAISS `store.add` / visual / ColPali build |
| Wrong or empty **answers**; bad citations | `main.run_rag_query` → `retrieval/pipeline.run_phase3_retrieval` → `build_grounded_prompt` → `answer_query` |
| **Retrieval** feels wrong (missing chunks, odd ranking) | Same as above; also check `SPARSE_INDEX_PATH` exists if hybrid is on, and `TOP_K` / `HYBRID_*` |
| **Vertex / GCP** errors | `generation/llm_pipeline.py` (`GeminiClient`), credentials in `.env` (`GOOGLE_APPLICATION_CREDENTIALS`, `GCP_PROJECT_ID`) |
| **Torch / CLIP / ColPali** crash or hang | `retrieval/visual_embedder.py`, `retrieval/colpali_retrieval.py`, `retrieval/torch_device.py`; try disabling flags (see §4) |
| **Streamlit** only | `ui/app.py` — confirm session state, paths to index files, and that `run_rag_query` arguments match CLI expectations |

---

## 2. Best default strategy: bisect with flags (no code change)

Most behavior is gated by **environment variables** (see [`.env.example`](../.env.example)). Copy to `.env` and turn **off** optional layers until the problem disappears; that tells you which subsystem to inspect.

**Suggested order (from cheap to expensive / broad):**

1. `ENABLE_QUERY_REFINEMENT=false` — removes extra LLM + second retrieval pass  
2. `ENABLE_COLPALI_RETRIEVAL=false` (and/or `ENABLE_COLPALI_INDEX=false` for ingest-only issues)  
3. `ENABLE_VISUAL_RETRIEVAL=false` — removes CLIP query + visual fusion  
4. `ENABLE_MULTI_HOP=false` — single retrieval hop  
5. `ENABLE_SEMANTIC_CACHE=false` — avoids stale cached answers while debugging retrieval  
6. `ENABLE_RERANK=false` — large change in ordering; use to see if reranker or model download is the issue  
7. `ENABLE_HYBRID=false` — dense-only; isolates BM25 / `SPARSE_INDEX_PATH` problems  

After you know which flag flips the bug, open the **## What this phase uses** section in the matching spec under [`docs/superpowers/specs/`](superpowers/specs/) for the exact modules and files.

---

## 3. Reproduce outside the UI

- **Tests:** `pytest -q` or narrow with `pytest tests/test_phase3_pipeline.py -v` / `pytest -k multihop -v`. Failing tests give a **fixed** reproduction.  
- **Minimal script:** From repo root, a short Python snippet that loads `Settings`, `FaissVectorStore.load(...)`, `Embedder`, and calls `run_rag_query(...)` removes Streamlit variables.  
- **Logging:** Run with `LOGLEVEL` / standard library — many paths use `logger = logging.getLogger(__name__)` (e.g. in `main.py`) with `warning` on skipped optional steps. For more noise:

```bash
export PYTHONWARNINGS=default
python -c "import logging; logging.basicConfig(level=logging.DEBUG)"  # then invoke your entrypoint
```

Wire `logging.basicConfig(level=logging.DEBUG)` at the top of a one-off script if needed.

---

## 4. Use the orchestrator as a map

Almost all query-time logic flows through **`main.run_rag_query`**. Set a **breakpoint** (IDE) or temporary `print()` / `breakpoint()` at:

1. After `effective_query` and modality router  
2. After refinement (if enabled)  
3. Before and after `_retrieve_effective` / `merge_dual_retrieval_contexts`  
4. After `run_phase3_retrieval` returns (inside `_retrieve_context` if you need deeper)  
5. Before `answer_query` — inspect **`context`** list: `chunk_id`, `score`, `modality`, `text` preview  

For **indexing**, breakpoint **`main.index_pdf`** and step into table/image/visual/ColPali branches.

---

## 5. Observability (optional)

If `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set, **`_trace_event`** in `main.py` sends lightweight traces (e.g. `run_rag_query` with `cache_hit`, `multi_hop`, `sub_query`, `refined_query`). Silent failure is intentional (`except: pass`); if nothing appears, verify keys and host.

---

## 6. Common failure modes (quick checks)

| Issue | Check |
|-------|--------|
| `ModuleNotFoundError` / wrong venv | Activate project `.venv`; `pip install -r requirements.txt` |
| FAISS dim mismatch | Embedding model at index time **same** as query time (`EMBEDDING_MODEL`); re-index if changed |
| Hybrid returns nothing useful | `python scripts/index_phase3_sparse.py` so `SPARSE_INDEX_PATH` matches current index metadata |
| Gemini 403 / auth | Service account JSON path, Vertex API enabled, `GCP_PROJECT_ID` / `GCP_LOCATION` |
| CLIP / ColPali segfault (macOS) | See README Phase 6 notes; disable visual/ColPali; single-threaded / MPS settings in code paths |
| Empty `context` | `TOP_K`, router intent, or index empty — print `store.ntotal` after load |

---

## 7. When to read vs when to change code

- **Read first:** phase **spec** + **plan** for the subsystem ([`docs/superpowers/README.md`](superpowers/README.md)), then the files listed in **## What this phase uses**.  
- **Change code last:** prefer a **minimal repro test** under `tests/` that mocks Vertex or FAISS so the fix is provable.

---

## 8. Related docs

- [ONBOARDING.md](ONBOARDING.md) — how to learn the repo structure  
- [superpowers/README.md](superpowers/README.md) — all phase specs and plans  

**Summary:** reproduce → **bisect with `.env` flags** → trace **`main.run_rag_query`** or **`index_pdf`** → use **tests** and **logging** → read the one phase doc for the layer you isolated.
