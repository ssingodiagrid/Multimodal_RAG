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

Edit `.env` (start from **`.env.example`** — it stays in git; `.env` is local):

- **Minimum for the app:** `GCP_PROJECT_ID`, `GCP_LOCATION`, `GEMINI_MODEL`, and **one** GCP credential approach (see table below).
- **Full catalog:** every variable the app and scripts read from the environment is listed in the **Environment variables** section below and mirrored in **`.env.example`**.

The app loads `.env` automatically if `python-dotenv` is installed (`requirements.txt` includes it).

**Security:** Do not commit `gcp-key.json` or `.env`. They are listed in `.gitignore`. If a key was ever committed or shared, **rotate it in GCP**.

### Environment variables (reference)

Copy `cp .env.example .env`, then set values. **Authoritative names and defaults** live in **`.env.example`**; this table summarizes purpose.

**Text index & chunking**

| Variable | Purpose |
|----------|---------|
| `CHUNK_SIZE` | Max characters per text chunk (recursive splitter target size). |
| `CHUNK_OVERLAP` | Overlap between consecutive text chunks. |
| `TOP_K` | Chunks passed to the generator after retrieval (and related caps). |
| `EMBEDDING_MODEL` | `sentence-transformers` model id for dense vectors (e.g. BGE-small). |
| `FAISS_INDEX_PATH` | Base path for the text FAISS index (`.faiss` + `.meta.json` siblings). |

**Vertex AI (required for generation / captions / LLM helpers)**

| Variable | Purpose |
|----------|---------|
| `GCP_PROJECT_ID` | GCP project id. |
| `GCP_LOCATION` | Vertex region (e.g. `us-central1`). |
| `GEMINI_MODEL` | Vertex **model resource** id (e.g. `gemini-2.0-flash-001`). |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to service account JSON (recommended). |
| `GOOGLE_APPLICATION_CREDENTIALS_DIR` | Directory containing the key file (with `GCP_KEY_FILENAME`). |
| `GCP_KEY_FILENAME` | Key file name inside `GOOGLE_APPLICATION_CREDENTIALS_DIR` (default `gcp-key.json`). |
| `GCP_SERVICE_ACCOUNT_KEY_PATH` | Explicit path to service account JSON. |

**Observability (optional)**

| Variable | Purpose |
|----------|---------|
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key (empty = no traces). |
| `LANGFUSE_SECRET_KEY` | Langfuse secret key. |
| `LANGFUSE_HOST` | Langfuse API host (default cloud host if unset). |

**Phase 2 evaluation**

| Variable | Purpose |
|----------|---------|
| `EVAL_DATASET_PATH` | JSONL eval questions/answers for `run_phase2_eval.py`. |
| `EVAL_OUTPUT_DIR` | Directory for eval JSON reports. |
| `PHASE2_EVAL_SOURCE_PDF` | Source PDF for `prepare_phase2_dataset.py`. |
| `PHASE2_CORPUS_PDF` | Corpus PDF for `scripts/index_phase2_corpus.py` (optional; script has its own default). |

**Phase 3 hybrid + rerank**

| Variable | Purpose |
|----------|---------|
| `ENABLE_HYBRID` | Dense + BM25 hybrid on/off. |
| `ENABLE_RERANK` | Cross-encoder rerank on/off. |
| `HYBRID_ALPHA` | Weight for dense vs sparse in fusion. |
| `DENSE_TOP_K` | FAISS candidate depth. |
| `SPARSE_TOP_K` | BM25 candidate depth. |
| `HYBRID_TOP_N` | Fused pool size before rerank. |
| `RERANK_TOP_K` | Reranker pool / cap. |
| `RERANK_MODEL` | Cross-encoder Hugging Face model id. |
| `SPARSE_INDEX_PATH` | BM25 JSON corpus path (from `index_phase3_sparse.py`). |

**Phase 4 cache, multi-hop, query refinement**

| Variable | Purpose |
|----------|---------|
| `ENABLE_SEMANTIC_CACHE` | Semantic query cache on/off. |
| `SEMANTIC_CACHE_THRESHOLD` | Min cosine similarity for cache hit. |
| `SEMANTIC_CACHE_MAX_ENTRIES` | Cache size cap / eviction. |
| `SEMANTIC_CACHE_PATH` | Cache files base path. |
| `ENABLE_MULTI_HOP` | Two-hop retrieval on/off. |
| `MULTI_HOP_MODE` | `off` \| `heuristic` \| `always`. |
| `MULTI_HOP_MERGED_TOP_K` | Max chunks after merging hop 1 + hop 2. |
| `ENABLE_QUERY_REFINEMENT` | LLM query rewrite + dual retrieval merge on/off. |

**Phase 5 tables, images, router**

| Variable | Purpose |
|----------|---------|
| `ENABLE_TABLE_EXTRACTION` | Extract and index tables. |
| `ENABLE_IMAGE_CAPTIONS` | Extract images + Gemini captions into index. |
| `ENABLE_IMAGE_PAGE_RENDERS` | Rasterize pages / figure regions for extra image chunks. |
| `IMAGE_PAGE_RENDER_DPI` | Render resolution. |
| `IMAGE_PAGE_RENDER_STRATEGY` | `figures` \| `full_pages`. |
| `ENABLE_MODALITY_ROUTER` | Route/bias retrieval by text/table/image intent. |
| `ROUTER_USE_LLM` | Use Gemini for router instead of heuristics only. |
| `TABLE_MAX_ROWS_PER_CHUNK` | Cap rows serialized per table chunk. |
| `IMAGE_CAPTION_MAX_SIDE` | Max image side (px) for captioning. |
| `ASSETS_DIR` | On-disk image (and related) assets per `doc_id`. |
| `TABLE_EXTRACTOR` | `pdfplumber` \| `camelot`. |

**Phase 6 CLIP visual index**

| Variable | Purpose |
|----------|---------|
| `ENABLE_VISUAL_RETRIEVAL` | Build/query CLIP visual FAISS on/off. |
| `VISUAL_FAISS_INDEX_PATH` | Visual FAISS base path. |
| `VISUAL_EMBEDDING_MODEL` | CLIP-class model id. |
| `VISUAL_TOP_K` | Visual search depth. |
| `VISUAL_FUSION_LAMBDA` | Fusion weight (text vs visual). |
| `VISUAL_DEVICE` | `mps` \| `cuda` \| `cpu` or empty = auto. |
| `VISUAL_BATCH_SIZE` | Batch size for visual encoding at index time. |
| `VISUAL_FOR_IMAGE_INTENT_ONLY` | Skip CLIP query encode for pure-text intent when true. |

**ColPali (full-page late interaction)**

| Variable | Purpose |
|----------|---------|
| `ENABLE_COLPALI_INDEX` | Build ColPali page index at PDF ingest. |
| `ENABLE_COLPALI_RETRIEVAL` | Run ColPali search at query time. |
| `COLPALI_MODEL_ID` | Hugging Face ColPali model id. |
| `COLPALI_INDEX_DIR` | On-disk ColPali index root. |
| `COLPALI_PAGE_DPI` | Raster DPI for pages. |
| `COLPALI_MAX_INDEX_PAGES` | Max pages to index (`0` = all). |
| `COLPALI_TOP_K` | Pages retrieved for the prompt. |
| `COLPALI_DEVICE` | `mps` \| `cuda` \| `cpu` or empty = auto. |
| `COLPALI_MAX_IMAGES_FOR_LLM` | Max page images sent to Gemini. |

When you add a new setting in `configs/settings.py`, update **`.env.example`** and this section so they stay in sync.

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

This uses a default corpus path inside the script unless you set **`PHASE2_CORPUS_PDF`** in `.env` (see environment table above).

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

Main controls are listed under **Setup → Environment variables** (Phase 3 table).

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

