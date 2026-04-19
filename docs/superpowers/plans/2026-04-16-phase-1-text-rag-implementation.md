# Phase 1 Text RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fully working text-only RAG pipeline (PDF -> chunks -> embeddings -> retrieval -> grounded answer + citations) with a minimal Streamlit UI and trace logging.

**Architecture:** Implement focused modules by capability (`ingestion`, `retrieval`, `generation`, `ui`) with shared schemas in `utils/schemas.py`. Keep FAISS as the local default, expose interfaces that allow Qdrant migration later without changing query orchestration code.

**Tech Stack:** Python, PyMuPDF, sentence-transformers (BGE-small), FAISS, Streamlit, Langfuse, pytest.

## What this phase uses

| Category | Items |
|----------|--------|
| **Runtime** | Python 3.10+ |
| **PDF / text** | **PyMuPDF** (`pymupdf`) — page text in `ingestion/pdf_loader.py`; chunking in `ingestion/chunker.py` |
| **Embeddings & index** | **sentence-transformers** — default `BAAI/bge-small-en-v1.5` in `retrieval/embedder.py`; **faiss-cpu** + metadata in `retrieval/vector_store.py` |
| **Generation** | **google-cloud-aiplatform** (Vertex AI) — `generation/llm_pipeline.py` (`GeminiClient`); `generation/prompt_builder.py` for grounded prompts |
| **UI** | **Streamlit** — `ui/app.py` |
| **Config** | `configs/settings.py`, `.env.example` (`CHUNK_*`, `TOP_K`, `EMBEDDING_MODEL`, `FAISS_INDEX_PATH`, `GCP_*`, `GEMINI_MODEL`, credentials) |
| **Observability (optional)** | **langfuse** — tracing hooks where wired |
| **Tests** | **pytest** — `tests/test_chunker.py`, `tests/test_vector_store.py`, `tests/test_retriever.py` |
| **Artifacts** | FAISS index + sidecar metadata under `data/parsed/` (paths from settings) |

Later phases reuse this stack; Phase 1 does **not** use BM25, semantic cache, or multimodal extractors.

---

## Scope Decomposition Note

The full Phase 1-6 roadmap contains multiple independent subsystems (evaluation, hybrid retrieval, caching, multimodal pipelines, late interaction retrieval). This plan intentionally covers **Phase 1 only** so implementation remains testable and shippable in one cycle. Subsequent phases should be delivered as separate plan files.

## File Structure (Phase 1)

- Create: `requirements.txt`
- Create: `.env.example`
- Create: `configs/settings.py`
- Create: `utils/schemas.py`
- Create: `ingestion/pdf_loader.py`
- Create: `ingestion/text_cleaner.py`
- Create: `ingestion/chunker.py`
- Create: `retrieval/embedder.py`
- Create: `retrieval/vector_store.py`
- Create: `retrieval/retriever.py`
- Create: `generation/prompt_builder.py`
- Create: `generation/llm_pipeline.py`
- Create: `ui/app.py`
- Create: `main.py`
- Create: `tests/test_chunker.py`
- Create: `tests/test_vector_store.py`
- Create: `tests/test_retriever.py`
- Create: `README.md`

### Task 1: Project Bootstrap and Config

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `configs/settings.py`
- Test: `tests/test_vector_store.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vector_store.py
from configs.settings import Settings

def test_settings_defaults():
    s = Settings()
    assert s.chunk_size == 800
    assert s.chunk_overlap == 100
    assert s.top_k == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_vector_store.py::test_settings_defaults -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'configs'`

- [ ] **Step 3: Write minimal implementation**

```python
# configs/settings.py
from dataclasses import dataclass
import os

@dataclass
class Settings:
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    faiss_index_path: str = os.getenv("FAISS_INDEX_PATH", "data/parsed/faiss.index")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_vector_store.py::test_settings_defaults -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_vector_store.py configs/settings.py requirements.txt .env.example
git commit -m "chore: bootstrap settings and environment defaults for phase 1"
```

### Task 2: PDF Loading and Text Cleaning

**Files:**
- Create: `ingestion/pdf_loader.py`
- Create: `ingestion/text_cleaner.py`
- Test: `tests/test_chunker.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chunker.py
from ingestion.text_cleaner import clean_text

def test_clean_text_collapses_whitespace():
    raw = "Revenue   increased\n\n\nin FY24."
    cleaned = clean_text(raw)
    assert cleaned == "Revenue increased in FY24."
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_chunker.py::test_clean_text_collapses_whitespace -v`  
Expected: FAIL with `ModuleNotFoundError` for `ingestion.text_cleaner`

- [ ] **Step 3: Write minimal implementation**

```python
# ingestion/text_cleaner.py
import re

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text
```

```python
# ingestion/pdf_loader.py
import fitz
from ingestion.text_cleaner import clean_text

def extract_pages(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc, start=1):
        pages.append({"page": i, "text": clean_text(page.get_text("text"))})
    return pages
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_chunker.py::test_clean_text_collapses_whitespace -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ingestion/pdf_loader.py ingestion/text_cleaner.py tests/test_chunker.py
git commit -m "feat: add pdf page extraction and text normalization"
```

### Task 3: Chunking with Metadata

**Files:**
- Create: `ingestion/chunker.py`
- Modify: `tests/test_chunker.py`
- Test: `tests/test_chunker.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chunker.py
from ingestion.chunker import chunk_pages

def test_chunk_pages_returns_metadata():
    pages = [{"page": 1, "text": "A " * 500}]
    chunks = chunk_pages("doc_1", pages, chunk_size=120, chunk_overlap=20)
    assert len(chunks) > 1
    assert chunks[0]["doc_id"] == "doc_1"
    assert "chunk_id" in chunks[0]
    assert chunks[0]["page"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_chunker.py::test_chunk_pages_returns_metadata -v`  
Expected: FAIL with `ModuleNotFoundError` for `ingestion.chunker`

- [ ] **Step 3: Write minimal implementation**

```python
# ingestion/chunker.py
def chunk_pages(doc_id: str, pages: list[dict], chunk_size: int = 800, chunk_overlap: int = 100) -> list[dict]:
    chunks = []
    step = max(chunk_size - chunk_overlap, 1)
    for page_obj in pages:
        text = page_obj["text"]
        page = page_obj["page"]
        i = 0
        chunk_num = 0
        while i < len(text):
            chunk_text = text[i:i + chunk_size]
            if chunk_text.strip():
                chunks.append({
                    "chunk_id": f"{doc_id}_p{page}_c{chunk_num}",
                    "doc_id": doc_id,
                    "page": page,
                    "text": chunk_text.strip(),
                })
            i += step
            chunk_num += 1
    return chunks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_chunker.py -v`  
Expected: PASS for both cleaner and chunker tests

- [ ] **Step 5: Commit**

```bash
git add ingestion/chunker.py tests/test_chunker.py
git commit -m "feat: add token-window style chunking with provenance metadata"
```

### Task 4: Embedding and Vector Store

**Files:**
- Create: `retrieval/embedder.py`
- Create: `retrieval/vector_store.py`
- Modify: `tests/test_vector_store.py`
- Test: `tests/test_vector_store.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vector_store.py
from retrieval.vector_store import InMemoryVectorStore

def test_add_and_search_vectors():
    store = InMemoryVectorStore()
    store.add([[1.0, 0.0], [0.0, 1.0]], [{"chunk_id": "a"}, {"chunk_id": "b"}])
    out = store.search([1.0, 0.0], top_k=1)
    assert out[0]["metadata"]["chunk_id"] == "a"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_vector_store.py::test_add_and_search_vectors -v`  
Expected: FAIL with `ModuleNotFoundError` for `retrieval.vector_store`

- [ ] **Step 3: Write minimal implementation**

```python
# retrieval/vector_store.py
import math

class InMemoryVectorStore:
    def __init__(self):
        self._vectors = []
        self._meta = []

    def add(self, vectors, metadata):
        self._vectors.extend(vectors)
        self._meta.extend(metadata)

    def search(self, query_vector, top_k=5):
        scored = []
        for v, m in zip(self._vectors, self._meta):
            dot = sum(a * b for a, b in zip(query_vector, v))
            nq = math.sqrt(sum(a * a for a in query_vector))
            nv = math.sqrt(sum(a * a for a in v))
            score = dot / (nq * nv + 1e-9)
            scored.append({"score": score, "metadata": m})
        return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]
```

```python
# retrieval/embedder.py
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_vector_store.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add retrieval/embedder.py retrieval/vector_store.py tests/test_vector_store.py
git commit -m "feat: add embedding interface and vector similarity store abstraction"
```

### Task 5: Retriever and Prompt Builder

**Files:**
- Create: `retrieval/retriever.py`
- Create: `generation/prompt_builder.py`
- Create: `utils/schemas.py`
- Modify: `tests/test_retriever.py`
- Test: `tests/test_retriever.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_retriever.py
from retrieval.retriever import retrieve_context

def test_retrieve_context_returns_ranked_chunks():
    store_results = [
        {"score": 0.9, "metadata": {"chunk_id": "c1", "text": "Revenue grew", "page": 1}},
        {"score": 0.8, "metadata": {"chunk_id": "c2", "text": "Costs rose", "page": 2}},
    ]
    context = retrieve_context(store_results, top_k=1)
    assert len(context) == 1
    assert context[0]["chunk_id"] == "c1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_retriever.py::test_retrieve_context_returns_ranked_chunks -v`  
Expected: FAIL with missing module/function

- [ ] **Step 3: Write minimal implementation**

```python
# retrieval/retriever.py
def retrieve_context(search_results: list[dict], top_k: int = 5) -> list[dict]:
    context = []
    for item in search_results[:top_k]:
        meta = item["metadata"]
        context.append({
            "chunk_id": meta["chunk_id"],
            "text": meta["text"],
            "page": meta["page"],
            "score": item["score"],
        })
    return context
```

```python
# generation/prompt_builder.py
def build_grounded_prompt(query: str, context_chunks: list[dict]) -> str:
    context_text = "\n".join([f"[p{c['page']}] {c['text']}" for c in context_chunks])
    return (
        "You are a grounded assistant.\n"
        "Use only the provided context.\n"
        "If the answer is not in context, respond with: I don't know.\n\n"
        f"Question: {query}\n\n"
        f"Context:\n{context_text}\n\n"
        "Answer with citations like (pX)."
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_retriever.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add retrieval/retriever.py generation/prompt_builder.py tests/test_retriever.py
git commit -m "feat: add retrieval context shaping and grounded prompt builder"
```

### Task 6: LLM Pipeline and End-to-End Orchestration

**Files:**
- Create: `generation/llm_pipeline.py`
- Create: `main.py`
- Test: `tests/test_retriever.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_retriever.py
from generation.prompt_builder import build_grounded_prompt

def test_prompt_contains_query_and_context():
    p = build_grounded_prompt("What changed?", [{"text": "Net income increased", "page": 3, "chunk_id": "x"}])
    assert "What changed?" in p
    assert "(p" in p or "[p3]" in p
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_retriever.py::test_prompt_contains_query_and_context -v`  
Expected: FAIL before `prompt_builder` exists, then PASS after Task 5; if already PASS, proceed.

- [ ] **Step 3: Write minimal implementation**

```python
# generation/llm_pipeline.py
import os
import google.generativeai as genai

class GeminiClient:
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(model_name)

    def answer(self, prompt: str) -> str:
        resp = self.model.generate_content(prompt)
        return resp.text
```

```python
# main.py
from generation.prompt_builder import build_grounded_prompt
from generation.llm_pipeline import GeminiClient

def answer_query(query: str, context_chunks: list[dict]) -> str:
    prompt = build_grounded_prompt(query, context_chunks)
    llm = GeminiClient()
    return llm.answer(prompt)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_retriever.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add generation/llm_pipeline.py main.py tests/test_retriever.py
git commit -m "feat: add gemini response pipeline and query answer orchestration"
```

### Task 7: Streamlit UI and Trace Integration

**Files:**
- Create: `ui/app.py`
- Modify: `README.md`
- Test: `tests/test_retriever.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_retriever.py
from generation.prompt_builder import build_grounded_prompt

def test_grounding_instruction_present():
    p = build_grounded_prompt("Q", [{"text": "A", "page": 1, "chunk_id": "1"}])
    assert "Use only the provided context" in p
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_retriever.py::test_grounding_instruction_present -v`  
Expected: FAIL until prompt builder is in place, then PASS.

- [ ] **Step 3: Write minimal implementation**

```python
# ui/app.py
import streamlit as st
from main import answer_query

st.title("Phase 1 Text RAG")
query = st.text_input("Ask a question")

if query:
    # placeholder context; replace with live retriever wiring in same task branch
    context = [{"chunk_id": "demo", "text": "Demo context", "page": 1, "score": 1.0}]
    answer = answer_query(query, context)
    st.subheader("Answer")
    st.write(answer)
    st.subheader("Retrieved Context")
    st.json(context)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_retriever.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ui/app.py README.md
git commit -m "feat: add streamlit query interface with answer and context display"
```

### Task 8: Verification and Developer Runbook

**Files:**
- Modify: `README.md`
- Test: `tests/test_chunker.py`, `tests/test_vector_store.py`, `tests/test_retriever.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chunker.py
def test_chunk_overlap_not_zero():
    from ingestion.chunker import chunk_pages
    pages = [{"page": 1, "text": "x" * 2000}]
    chunks = chunk_pages("doc", pages, chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_chunker.py::test_chunk_overlap_not_zero -v`  
Expected: FAIL if chunking not implemented as planned.

- [ ] **Step 3: Write minimal implementation**

```markdown
# README.md
## Run
1. `python -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. `cp .env.example .env` and set `GOOGLE_API_KEY`
4. `pytest -v`
5. `streamlit run ui/app.py`

## Expected behavior
- Enter a query and receive grounded answer.
- Retrieved context is shown with page metadata.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_chunker.py tests/test_vector_store.py tests/test_retriever.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add README.md tests/test_chunker.py
git commit -m "docs: add phase 1 setup, test, and runbook instructions"
```

## Verification Checklist (Before Marking Phase 1 Complete)

- `pytest tests/test_chunker.py tests/test_vector_store.py tests/test_retriever.py -v` passes.
- `streamlit run ui/app.py` starts and answers sample query.
- Retrieval output includes page-level metadata.
- Prompt includes explicit abstention rule.
- `.env.example` documents required keys and defaults.

## Next Plan Files To Create (After This Plan Is Executed)

- `docs/superpowers/plans/2026-04-16-phase-2-evaluation-implementation.md`
- `docs/superpowers/plans/2026-04-16-phase-3-hybrid-rerank-implementation.md`
- `docs/superpowers/plans/2026-04-16-phase-4-cache-multihop-implementation.md`
- `docs/superpowers/plans/2026-04-16-phase-5-multimodal-practical-implementation.md`
- `docs/superpowers/plans/2026-04-16-phase-6-advanced-multimodal-implementation.md`
