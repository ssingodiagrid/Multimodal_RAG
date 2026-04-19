# Multimodal RAG — Phase 1 (Text RAG)

Text-only pipeline: PDF → chunks → embeddings (BGE-small) → FAISS → **Gemini on Vertex AI** answers with citations.

**New to the repo?** Read [docs/ONBOARDING.md](docs/ONBOARDING.md) for a guided path through the markdown docs, `main.py`, and tests.

**Debugging?** See [docs/DEBUGGING.md](docs/DEBUGGING.md) — env-flag bisection, trace `main.run_rag_query` / `index_pdf`, pytest, Langfuse, common failures.

## Setup

1. Create a virtual environment and install dependencies:

```bash
cd /Users/ssingodia/Desktop/RAG
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Copy environment template and configure **Vertex AI**:

```bash
cp .env.example .env
```

Edit `.env`:

- `GCP_PROJECT_ID` — your GCP project
- `GCP_LOCATION` — e.g. `us-central1`
- `GEMINI_MODEL` — Vertex model id, e.g. `gemini-2.0-flash-001`
- **Credentials (pick one):**
  - Set `GOOGLE_APPLICATION_CREDENTIALS` to the path of your service account JSON (e.g. `./gcp-key.json`), **or**
  - Set `GOOGLE_APPLICATION_CREDENTIALS_DIR` (e.g. `./secrets`) and place `gcp-key.json` there (or set `GCP_KEY_FILENAME`), **or**
  - Set `GCP_SERVICE_ACCOUNT_KEY_PATH` to the JSON path

The app loads `.env` automatically if `python-dotenv` is installed (`requirements.txt` includes it).

**Security:** Do not commit `gcp-key.json` or `.env`. They are listed in `.gitignore`. If a key was ever committed or shared, **rotate it in GCP**.

3. Enable the **Vertex AI API** for your project and ensure the service account has permission to use Generative AI on Vertex (e.g. `Vertex AI User`).

4. Run tests:

```bash
pytest tests/test_chunker.py tests/test_vector_store.py tests/test_retriever.py -v
```

**Note:** First run of embedding tests or the app may download `BAAI/bge-small-en-v1.5` (network required).

## Run the UI

```bash
source .venv/bin/activate
streamlit run ui/app.py
```

1. In the sidebar, upload a PDF and click **Index uploaded PDF**.
2. Ask a question; the app shows the answer and retrieved chunks (page metadata).

## Observability (optional)

Set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and optionally `LANGFUSE_HOST` in `.env`. Index and query paths emit best-effort Langfuse traces when the SDK is configured.

## Layout

- `ingestion/` — PDF text extraction, cleaning, chunking
- `retrieval/` — embeddings, FAISS store, context shaping
- `generation/` — grounded prompts, Vertex Gemini client
- `main.py` — index PDF, load index, end-to-end RAG query
- `ui/app.py` — Streamlit frontend

**Architecture** (end-to-end: what each library/module is for): [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

**Design docs** (specs + code maps): [`docs/superpowers/README.md`](docs/superpowers/README.md)

## Expected behavior

- Answers should cite pages like `(p3)` when the model follows the prompt.
- If the answer is not in the context, the model should say it does not know (per prompt).

## Phase 2 Evaluation

Index the evaluation corpus first (your IFC annual report):

```bash
python scripts/index_phase2_corpus.py
```

This uses `/Users/ssingodia/Desktop/RAG/ifc-annual-report-2024-financials.pdf` by default.
Override with:

```bash
PHASE2_CORPUS_PDF="/absolute/path/to/your.pdf" python scripts/index_phase2_corpus.py
```

Prepare dataset from your provided source PDF:

```bash
python scripts/prepare_phase2_dataset.py
```

Run evaluation tests:

```bash
pytest tests/test_eval_dataset_loader.py tests/test_eval_metrics.py tests/test_eval_report_writer.py -v
```

Run end-to-end evaluation:

```bash
python scripts/run_phase2_eval.py
```

Expected output:
- JSON report in `data/eval/reports/`
- summary metrics printed in terminal (`answer_relevance`, `faithfulness`, `context_relevance`)

## Phase 3 Hybrid Retrieval + Reranking

Build sparse corpus artifact from the existing indexed metadata:

```bash
python scripts/index_phase3_sparse.py
```

Run Phase 3 tests:

```bash
pytest tests/test_bm25.py tests/test_hybrid_retriever.py tests/test_reranker.py tests/test_phase3_pipeline.py -v
```

Controls in `.env`:
- `ENABLE_HYBRID=true|false`
- `ENABLE_RERANK=true|false`
- `HYBRID_ALPHA`
- `SPARSE_INDEX_PATH`

## Phase 4 Semantic cache + multi-hop

**Semantic cache** stores query embedding, answer, and context under `SEMANTIC_CACHE_PATH` (FAISS + metadata). A **fingerprint** derived from the corpus FAISS file, optional sparse JSON, and embedding model name prevents cache hits after re-indexing or model changes. Set `ENABLE_SEMANTIC_CACHE=true` to enable. For Phase 2-style eval runs, leave it `false` so metrics stay comparable.

**Multi-hop** runs an extra retrieval pass: initial Phase 3 retrieval → Gemini emits a JSON `sub_query` → second retrieval → contexts merged and deduped by `chunk_id`. Set `ENABLE_MULTI_HOP=true` and tune `MULTI_HOP_MODE` (`off` | `heuristic` | `always`). Heuristic mode triggers on comparative phrasing (e.g. “compare”, “versus”, “difference between”).

**Query refinement (optional):** `ENABLE_QUERY_REFINEMENT=true` runs a short Gemini rewrite **before** retrieval, embeds **both** raw and rewritten queries, runs Phase 3 **twice**, and merges chunks by `chunk_id` (max score; optional `retrieval_source` on each chunk). Semantic cache is **off** while this flag is on. Skipped for query-image uploads. Code: `generation/query_refinement.py`, `retrieval/dual_query_merge.py`, `main.run_rag_query`, `generation/prompt_builder.py`. Docs: [design](docs/superpowers/specs/2026-04-17-phase-4-query-refinement-design.md) · [implementation / file map](docs/superpowers/plans/2026-04-17-phase-4-query-refinement-implementation.md).

Re-indexing from the Streamlit sidebar clears dense, sparse, and **semantic cache** artifacts.

Run Phase 4 tests:

```bash
pytest tests/test_phase4_settings.py tests/test_index_fingerprint.py tests/test_semantic_cache.py tests/test_multihop.py tests/test_phase4_orchestration.py tests/test_dual_query_merge.py -v
```

## Phase 5 Tables + images + modality router

**Indexing:** With `ENABLE_TABLE_EXTRACTION=true`, PDFs are parsed for tables (pdfplumber by default; set `TABLE_EXTRACTOR=camelot` if you install `camelot-py` and system deps). With `ENABLE_IMAGE_CAPTIONS=true`, embedded images are saved under `ASSETS_DIR/{doc_id}/` and captioned with **Vertex Gemini** (multimodal). Chunks are merged into the same FAISS + BM25 index as body text (`modality`: `text` | `table` | `image`).

**Retrieval:** `ENABLE_MODALITY_ROUTER=true` biases ranking toward table/image chunks when the query looks tabular or visual (heuristic); optional `ROUTER_USE_LLM=true` uses a short Gemini JSON classifier.

**UI:** Retrieved context shows **per-chunk** evidence: `st.dataframe` for `table_json`, `st.image` for `asset_path`, plus raw JSON in a nested expander. Re-indexing clears `ASSETS_DIR` along with dense/sparse/cache artifacts.

```bash
pytest tests/test_phase5_settings.py tests/test_tables_normalize.py tests/test_modality_router.py tests/test_modality_rank.py tests/test_prompt_builder_phase5.py tests/test_build_final_context_phase5.py -v
```

Optional: `pip install camelot-py[cv]` (Ghostscript) for Camelot lattice tables.

**Figure pixmaps:** With `ENABLE_IMAGE_PAGE_RENDERS=true` (and image captions on), PyMuPDF **page pixmaps** are indexed for vector figures; see `.env.example`.

## Phase 6 Visual retrieval (optional)

When `ENABLE_VISUAL_RETRIEVAL=true`, indexing also builds **`VISUAL_FAISS_INDEX_PATH`** (CLIP-class embeddings of image assets). At query time, for `image` / `mixed` router intents, visual hits are **fused** with the Phase 3 dense+hybrid list (`VISUAL_FUSION_LAMBDA`). Requires Phase 5 image chunks and files under `ASSETS_DIR`. See `.env.example`.

If the text index already exists but **`faiss.index.visual.faiss` is missing**, turn Phase 6 on in `.env` and run **`python scripts/rebuild_visual_faiss.py`** (uses existing `faiss.index.meta.json` + on-disk assets; first CLIP run may download weights).

**macOS:** If CLIP encoding **segfaults**, use the project **venv** (not mixed with another Python), keep only one of conda/system OpenMP in `DYLD_LIBRARY_PATH`, or run on a Linux machine / CI. The embedder sets `OMP_NUM_THREADS=1` and encodes **one image per forward** to avoid common native crashes.

**ColPali (optional late interaction):** `ENABLE_COLPALI_INDEX` / `ENABLE_COLPALI_RETRIEVAL` — full-page rasters, MaxSim page search (`retrieval/colpali_retrieval.py`), merged into `run_rag_query` context, multimodal answer in `answer_query`. Rebuild: `python scripts/rebuild_colpali.py`. Env knobs: `COLPALI_*` in `.env.example`.

Docs: [Phase 6 design](docs/superpowers/specs/2026-04-17-phase-6-visual-retrieval-design.md) · [implementation plan + file map](docs/superpowers/plans/2026-04-17-phase-6-visual-retrieval-implementation.md).

```bash
pytest tests/test_phase6_settings.py tests/test_visual_fusion.py tests/test_index_fingerprint.py tests/test_colpali_maxsim.py -v
```
# Multimodal_RAG
