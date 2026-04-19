# Multimodal RAG Phase 1-6 Design

## Goal

Build a production-style multimodal RAG platform that evolves from text-only retrieval to advanced visual late-interaction retrieval while preserving one stable system architecture, one set of interfaces, and repeatable evaluation across all phases.

## Scope

This design covers:

- Phase 1: text-only RAG baseline
- Phase 2: quality evaluation with RAGAS and LLM judge
- Phase 3: hybrid retrieval and reranking
- Phase 4: semantic cache and multi-hop retrieval; optional **pre-retrieval LLM query rewrite** with dual-vector retrieval merge ([`2026-04-17-phase-4-query-refinement-design.md`](./2026-04-17-phase-4-query-refinement-design.md))
- Phase 5: table and image-capable multimodal RAG (caption-based image retrieval)
- Phase 6: advanced multimodal retrieval with CLIP baseline and ColPali-style late interaction

Out of scope:

- Enterprise auth/SSO
- Multi-tenant billing
- Human annotation tooling beyond evaluation datasets

## Documentation convention

Per-phase **design** specs in `docs/superpowers/specs/` include a **## What this phase uses** section: libraries, dependencies on earlier phases, main Python modules, artifacts, and primary `.env` flags. Matching **implementation** plans in `docs/superpowers/plans/` include the same section (or a short pointer to the spec) so tooling and file paths stay aligned with the code.

## Architecture Overview

The system is built as a single query orchestration pipeline with phase-gated capability flags instead of separate disconnected implementations per phase.

High-level flow:

1. User query enters orchestrator (LangGraph state machine).
2. Query router classifies intent and modality needs (text/table/image/mixed).
3. Semantic cache checks for reusable answer.
4. Retrieval coordinator executes retrieval strategy (dense, sparse, hybrid, multimodal).
5. Re-ranker orders candidate evidence.
6. Context builder normalizes evidence and citations.
7. Generator (Gemini or Qwen2-VL) produces grounded answer.
8. Response returns with citations and retrieval metadata.
9. Trace + metrics are logged to observability and evaluation stores.

## Design Principles

- Progressive enhancement: every new phase layers on top of previous contracts.
- Stable interfaces: ingestion, retrieval, reranking, generation, and evaluation communicate through fixed schemas.
- Retrieval-first quality: prefer retrieval improvements before model swaps.
- Measurable progress: each phase has mandatory evaluation gates.
- Fallback safety: when advanced path fails, degrade to a lower-capability but valid path.

## Module Boundaries

### 1) Ingestion

Responsibilities:

- Parse PDFs into text blocks, tables, and image assets.
- Attach normalized metadata (doc_id, page, section, bbox, modality).
- Produce phase-compatible artifacts and index payloads.

Inputs:

- Raw PDF files in `data/raw_pdf/`

Outputs:

- `data/parsed/text_chunks.jsonl`
- `data/tables/tables.jsonl` and CSV extracts
- `data/images/*` and `data/parsed/image_captions.jsonl`
- `data/parsed/doc_manifest.json`

### 2) Retrieval

Responsibilities:

- Execute dense retrieval (embedding vector search).
- Execute sparse retrieval (BM25).
- Run hybrid score fusion.
- Route to multimodal retrieval when needed.

Inputs:

- Query + optional metadata filters

Outputs:

- Candidate evidence list with modality, score, and provenance

### 3) Reranking

Responsibilities:

- Cross-encoder ranking over top-N retrieved candidates.
- Return compact top-K list for generation.

### 4) Generation

Responsibilities:

- Build grounded prompt/context packet.
- Generate answer and confidence notes.
- Enforce citation grounding and abstention policy.

### 5) Advanced Runtime Features

Responsibilities:

- Semantic cache lookup/store.
- Multi-hop query decomposition and iterative retrieval.

### 6) Evaluation

Responsibilities:

- Run phase-specific benchmarks and regression suites.
- Compute RAGAS and judge scores.
- Persist snapshots for phase-over-phase comparison.

### 7) Observability

Responsibilities:

- End-to-end trace logging for ingestion/query/evaluation.
- Latency, retrieval depth, token/cost stats, cache hit stats.

## Canonical Data Contracts

### DocumentUnit

- `id`: unique unit id
- `doc_id`: source document
- `page`: page number
- `modality`: `text|table|image|patch`
- `content`: normalized textual payload (if applicable)
- `asset_ref`: pointer to image/table asset if applicable
- `metadata`: section, bbox, tags, extraction_confidence

### RetrievalResult

- `unit_id`
- `score`
- `retriever`: dense|sparse|hybrid|multimodal
- `modality`
- `source_ref`: doc/page/region
- `debug`: component scores and rank positions

### ContextPacket

- `query`
- `evidence[]`: ordered retrieval results
- `citations[]`
- `reasoning_hints`: optional structured hints for generator

### EvalRecord

- `query`
- `expected_answer`
- `retrieved_context`
- `model_answer`
- `ragas_metrics`
- `judge_metrics`
- `phase_tag`
- `run_id`

## Phase-by-Phase Capability Plan

Each phase below is intentionally descriptive and operational: what is done, what is used, and the exact step sequence to follow.

### Phase 1: Text RAG Foundation

**What will be done**

- Build a complete text-only RAG pipeline from PDF ingestion to grounded answer generation.
- Establish canonical document/chunk metadata that later phases reuse.
- Provide a minimal UI to inspect answers and retrieved evidence.

**What will be used**

- Extraction: PyMuPDF (or pdfminer fallback)
- Chunking: token-aware chunker (500-1000 tokens, 100 overlap)
- Embeddings: BGE-small (or Gemini embeddings, configurable)
- Vector DB: FAISS for local, Qdrant-compatible abstraction
- Generation: Gemini 2.0 Flash
- UI/Tracing: Streamlit + Langfuse

**Steps to follow**

1. Implement text extraction and save page-linked text artifacts.
2. Clean and normalize text (headers/footers, whitespace, unicode normalization as needed).
3. Chunk text with overlap; attach `doc_id`, `page`, `section`, `chunk_id`.
4. Generate dense embeddings and store vectors with metadata.
5. Implement dense retriever (`top_k=5` baseline).
6. Build grounded prompt template: "answer only from context, abstain if unknown."
7. Implement answer generation with citation references.
8. Add Streamlit page for query, answer, and retrieved chunk preview.
9. Add tracing for each query path.

**Deliverables**

- Runnable text-only RAG pipeline
- Persisted text chunk index
- UI with evidence display

**Success criteria**

- End-to-end query answering works on project documents.
- Responses include source citations.

### Phase 2: Evaluation Baseline

**What will be done**

- Create a repeatable quality measurement layer for retrieval + generation.
- Define baseline metrics to compare all future phases.

**What will be used**

- RAGAS: context relevance, faithfulness, answer relevance
- LLM-as-judge: Gemini rubric scoring
- Storage: JSONL or parquet eval records with run IDs

**Steps to follow**

1. Load curated Q/A benchmark dataset and normalize schema.
2. Run pipeline for each benchmark query and capture retrieved context + answer.
3. Compute RAGAS metrics for each sample.
4. Run judge prompt for correctness, hallucination risk, and reasoning quality.
5. Aggregate scores into per-run summary and per-query diagnostics.
6. Save baseline report tagged `phase_2_baseline`.
7. Add regression command that reruns eval on any retrieval/prompt/model change.

**Deliverables**

- Evaluation scripts and report artifacts
- Baseline metrics snapshot used for phase comparisons

**Success criteria**

- Automated eval script outputs reproducible metrics report.
- Baseline scores are stored for future phase comparisons.

### Phase 3: Retrieval Quality Upgrade

**What will be done**

- Improve retrieval recall and precision by combining sparse and dense search.
- Improve final evidence quality before generation with reranking.

**What will be used**

- Sparse retrieval: BM25 (`rank_bm25`)
- Dense retrieval: existing embedding index
- Fusion: weighted score (`alpha*dense + (1-alpha)*sparse`, alpha tunable)
- Reranking: cross-encoder (BGE reranker or MiniLM cross-encoder)

**Steps to follow**

1. Build BM25 index from same chunk corpus used by dense retrieval.
2. Return candidate sets from both dense and sparse retrievers.
3. Implement score normalization and weighted fusion.
4. Retrieve top-20 fused candidates for reranking.
5. Apply cross-encoder reranker and keep top-5 evidence units.
6. Feed reranked evidence into generator unchanged.
7. Re-run Phase 2 eval suite and compare against baseline.
8. Tune `alpha`, candidate depth, and rerank top-N using eval data.

**Deliverables**

- Hybrid retriever service
- Reranking module
- Comparative eval report vs Phase 2

**Success criteria**

- Measurable improvement over Phase 2 baseline on retrieval-linked metrics.

### Phase 4: Runtime Optimization + Reasoning Depth

**What will be done**

- Reduce cost/latency for repeated queries using semantic caching.
- Improve compositional query handling using multi-hop retrieval.

**What will be used**

- Semantic cache: embedding similarity index (FAISS/Qdrant collection)
- Multi-hop controller: in-repo Python orchestrator (retrieve → sub-query → retrieve); LangGraph deferred to a future refactor
- **Sub-query refinement:** Gemini JSON `sub_query` for hop 2 (compositional questions)
- **Optional pre-retrieval rewrite:** Gemini search-query paraphrase + dual Phase 3 passes merged by `chunk_id` ([extension spec](./2026-04-17-phase-4-query-refinement-design.md))

**Steps to follow**

1. Add query embedding cache lookup before retrieval.
2. Define cache hit threshold and freshness/version constraints.
3. On hit: return cached answer + provenance immediately.
4. On miss: run normal retrieval pipeline and persist result to cache.
5. Add multi-hop mode for complex/comparative queries.
6. Hop 1: retrieve initial evidence; generate focused sub-query.
7. Hop 2: retrieve supplemental evidence; merge/deduplicate context.
8. Generate final answer using combined evidence.
9. Evaluate latency, hit rate, and multi-hop quality delta.

**Deliverables**

- Semantic cache service with configurable threshold
- Multi-hop retrieval path integrated into orchestrator
- Runtime performance report

**Success criteria**

- Cache hit path reduces latency/cost for repeated intents.
- Multi-hop questions improve answer quality compared to single-hop.

### Phase 5: Practical Multimodal Support

**What will be done**

- Add first-class handling for tables and images without full visual-token retrieval yet.
- Route modality-specific queries to dedicated table/image paths.

**What will be used**

- Table extraction: Camelot and/or Tabula
- Table processing: pandas + normalization to JSON/text rows
- Image extraction: PyMuPDF
- Image understanding: Gemini caption generation
- Retrieval: caption/text embedding index + existing hybrid text retriever

**Steps to follow**

1. Extract tables from PDFs and save structured outputs (CSV/JSON).
2. Normalize tables into retrieval-friendly text facts plus structured payload.
3. Index table facts with metadata (page, table_id, column names).
4. Extract document images/charts and persist image assets.
5. Generate captions/descriptions for images; store caption index entries.
6. Extend router to detect table/image intents and mixed intents.
7. For table queries: retrieve table evidence and pass structured context to generator.
8. For image queries: retrieve caption evidence and pass image metadata to generator.
9. Re-run evaluation with table/image question subsets.

**Deliverables**

- Table extraction + indexing pipeline
- Image captioning + retrieval pipeline
- Modality-aware query router

**Success criteria**

- System answers table and chart questions with source grounding.
- Modality router correctly selects text/table/image pipeline.

### Phase 6: Advanced Multimodal Retrieval

**What will be done**

- Upgrade from caption-only visual retrieval to true visual embedding retrieval.
- Add late-interaction retrieval for fine-grained document image/patch matching.
- Use multimodal generation for stronger chart/layout reasoning.

**What will be used**

- Baseline visual-text alignment: CLIP
- Advanced retrieval: ColPali-style late interaction (with Byaldi-style serving pattern if chosen)
- Multimodal generation: Qwen2-VL
- Evaluation: ViDoRe-style retrieval tasks + existing judge stack

**Steps to follow**

1. Convert PDF pages into image views and create patch/grid representations.
2. Build CLIP baseline index for text-to-image retrieval comparison.
3. Build ColPali-style index (token/patch-level representations).
4. Implement late-interaction scoring (MaxSim-style token-patch matching).
5. Return top visual regions/pages with confidence and provenance.
6. Feed retrieved visual evidence to Qwen2-VL for multimodal grounded answering.
7. Add region-level citation output (page + bbox/patch reference).
8. Evaluate against caption baseline using visual retrieval/QA benchmarks.
9. Keep caption-based fallback path for robustness.

**Deliverables**

- CLIP baseline multimodal retriever
- ColPali-style advanced multimodal retriever
- Qwen2-VL reasoning path with region attribution

**Success criteria**

- Outperforms caption-only approach on visual retrieval tasks.
- Produces grounded multimodal answers with attributable visual evidence.

## Routing and Fallback Strategy

1. Classify query intent and modality.
2. If cache hit above threshold, return cached answer.
3. For text-heavy queries: hybrid retrieval + rerank.
4. For table-heavy queries: table retriever + optional text join.
5. For image/visual queries:
   - Phase 5: caption index retrieval
   - Phase 6: CLIP/ColPali retrieval first, caption fallback
6. If multimodal retrieval confidence is low, fallback to text hybrid path and disclose uncertainty.

## Table and Image Extraction Design

This section defines exactly how PDF tables and images are extracted, normalized, stored, and indexed for multimodal retrieval.

### Table Extraction Pipeline

**Goal**

Convert raw PDF table layouts into both structured data and retrieval-friendly semantic units while preserving page-level provenance.

**Primary tools**

- Camelot (`lattice` and `stream` modes)
- Tabula fallback for difficult layouts
- pandas for normalization/cleanup

**Execution flow**

1. Iterate page-by-page through each input PDF.
2. Run Camelot in `lattice` mode first (good for ruled tables).
3. If low-quality parse or no table detected, run Camelot `stream` mode.
4. If still low-quality, run Tabula fallback extraction.
5. Convert extracted table into DataFrame and normalize:
   - header cleanup
   - merged-cell flattening where possible
   - numeric normalization (`$`, `%`, commas, brackets)
6. Write outputs:
   - structured artifact: CSV + JSON
   - semantic artifact: row/column fact strings for retrieval
7. Attach provenance metadata:
   - `doc_id`, `page`, `table_id`, `bbox` (if available), column names, extraction quality score
8. Index semantic table facts into retrieval index; keep structured artifact reference for final answer rendering.

**Quality checks**

- Reject table if empty cell ratio is above threshold.
- Reject table if detected header width is inconsistent with body for most rows.
- Mark uncertain tables for fallback text-only handling instead of hard failure.

### Image Extraction Pipeline

**Goal**

Convert document images/charts into indexed evidence with semantic descriptions and source-attribution-ready metadata.

**Primary tools**

- PyMuPDF for image extraction and region rendering
- Gemini for image captioning (Phase 5 baseline)
- Optional OCR pass for axis labels/embedded text

**Execution flow**

1. For each PDF page, extract embedded image objects via PyMuPDF.
2. If page contains chart-like vector content not exposed as embedded image, render region/page snapshot.
3. Persist each image with stable asset id (`doc_id`, `page`, `image_id`).
4. Generate caption with Gemini:
   - chart type
   - trend summary
   - key entities/time ranges if visible
5. Optionally run OCR to capture axis labels and short legends.
6. Store caption + OCR text as semantic retrieval content.
7. Attach provenance metadata:
   - `doc_id`, `page`, `image_id`, `bbox`/region reference, caption confidence
8. Index caption/OCR text into caption retrieval index (Phase 5).
9. In Phase 6, additionally index visual embeddings for CLIP/ColPali retrieval.

**Quality checks**

- Enforce minimum caption quality and retry once on low-confidence outputs.
- If caption generation fails, keep image artifact and mark for visual-only fallback path.

### Extraction Orchestration and Fallbacks

- Extraction is non-blocking by modality: table or image failures do not stop text ingestion.
- Every extracted unit is versioned by ingestion run id for reproducibility.
- Missing table/image extraction gracefully degrades to text retrieval with explicit confidence notes.

### Table/Image Flow Diagram

```text
PDF Document
   |
   +--> Text Path ----------------> chunk -> embed -> text index
   |
   +--> Table Path
   |      -> detect (Camelot lattice/stream)
   |      -> fallback (Tabula)
   |      -> normalize (pandas)
   |      -> save (CSV/JSON + semantic facts)
   |      -> index (table semantic index)
   |
   +--> Image Path
          -> extract/render (PyMuPDF)
          -> caption (Gemini) + optional OCR
          -> save assets + metadata
          -> index captions (Phase 5)
          -> index visual embeddings (Phase 6: CLIP/ColPali)
```

## Error Handling

- Extraction failures: mark artifact with failure metadata; continue partial ingestion.
- Missing modality index: fallback to available modalities.
- Empty retrieval: abstain with explicit "insufficient context."
- Low confidence rerank spread: trigger multi-hop (if enabled) or ask clarifying follow-up.
- Model timeout: retry once, then fallback model path.

## Testing Strategy

### Unit Tests

- Chunking boundaries and overlap behavior.
- Metadata integrity for each ingestion artifact type.
- Hybrid fusion score correctness.
- Router decision logic.
- Cache threshold semantics.

### Integration Tests

- End-to-end text query flow.
- End-to-end table query flow.
- End-to-end image query flow (caption path, then advanced path in Phase 6).
- Multi-hop flow with query decomposition.

### Regression/Evaluation Tests

- Phase-level benchmark suite with fixed dataset.
- Mandatory metric comparison against previous phase baseline.
- Track precision@k, mrr, groundedness, faithfulness, latency, token cost.

## Observability and Operations

- Trace every query with request id and phase capability flags.
- Log retriever candidate sets and reranker deltas for debugging.
- Emit latency breakdown: routing, retrieval, rerank, generation.
- Emit cache hit rate and fallback rate.
- Maintain index versioning so evaluation runs are reproducible.

## Security and Data Handling

- Store only required document artifacts and metadata.
- Keep API keys in environment variables.
- Mask sensitive fields in traces/logs when needed.
- Separate raw assets from derived indexes for safe reindexing.

## Suggested Folder Structure

```text
project/
  data/
    raw_pdf/
    parsed/
    tables/
    images/
  ingestion/
  retrieval/
  reranking/
  generation/
  evaluation/
  multimodal/
  cache/
  ui/
  configs/
  utils/
  docs/superpowers/specs/
  docs/superpowers/plans/
```

## Risks and Mitigations

- Risk: Jumping to Phase 6 too early.
  - Mitigation: Enforce phase gates with evaluation thresholds.
- Risk: Strong LLM masking weak retrieval.
  - Mitigation: Track retrieval metrics independently from generation scores.
- Risk: Multimodal complexity inflates latency/cost.
  - Mitigation: Capability flags + staged fallback and cache strategy.
- Risk: Poor table extraction on complex layouts.
  - Mitigation: dual extractor strategy (Camelot/Tabula) + LLM normalization.

## Definition of Done (Program-Level)

- All phases implemented behind stable orchestration interfaces.
- Phase-by-phase evaluation reports persisted and comparable.
- UI/API can answer text, table, and image questions with citations.
- Advanced multimodal path (Phase 6) can retrieve and reason over visual evidence.
- Observability captures full query lifecycle for debugging and optimization.

