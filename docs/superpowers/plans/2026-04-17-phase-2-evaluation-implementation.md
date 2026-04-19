# Phase 2 Evaluation (RAGAS + LLM Judge) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible evaluation pipeline that scores Phase 1 RAG quality using RAGAS-style metrics and a Vertex LLM-as-judge rubric, then stores run artifacts for comparison across future phases.

**Architecture:** Add an `evaluation/` module with clean boundaries: dataset loading, pipeline runner, metric computation, judge scoring, and report persistence. Keep the core app unchanged; evaluation is an offline command path.

**Tech Stack:** Python, pytest, pandas/json, Vertex AI Gemini, RAGAS (or fallback metric shim), existing Phase 1 retrieval/generation modules.

## What this phase uses

| Category | Items |
|----------|--------|
| **From Phase 1** | `main.run_rag_query`, `main.load_index` / indexing entrypoints, `Embedder`, `FaissVectorStore`, `GeminiClient` — eval **calls the real RAG path** (or mocks in unit tests) |
| **Libraries** | **pandas** — dataset tables; **ragas** — RAG-style metrics in `evaluation/ragas_eval.py`; **google-cloud-aiplatform** — judge / metric LLM calls via Vertex |
| **Project modules** | `evaluation/dataset_loader.py`, `dataset_preprocessor.py`, `eval_runner.py`, `llm_judge.py`, `report_writer.py` |
| **Scripts** | `scripts/prepare_phase2_dataset.py` — build JSONL; `scripts/run_phase2_eval.py` — batch eval |
| **Data layout** | `Settings.eval_dataset_path`, `eval_output_dir` (see `.env.example`: `EVAL_DATASET_PATH`, `EVAL_OUTPUT_DIR`) |
| **Tests** | `tests/test_eval_dataset_loader.py`, `tests/test_eval_metrics.py`, `tests/test_eval_report_writer.py` |

Phase 2 **does not** change hybrid retrieval, cache, or multimodal indexing; it **measures** the stack produced by Phase 1+.

---

## Dataset Source for Phase 2

Primary evaluation source file:

- `/Users/ssingodia/Desktop/RAG/RAG_evaluation_dataset - convertcsv (2).pdf`

This PDF must be converted into normalized JSONL before running evaluation. Target schema:

```json
{"query":"...","expected_answer":"...","context":"...","metadata":{"id":"..."}}
```

If `context` is unavailable in the source file, keep minimum schema:

```json
{"query":"...","expected_answer":"..."}
```

## Scope Decomposition Note

This plan covers **Phase 2 only**. It intentionally focuses on measurement, not retrieval architecture changes (Phase 3) or multimodal paths (Phase 5/6).

## File Structure (Phase 2)

- Create: `evaluation/__init__.py`
- Create: `evaluation/dataset_loader.py`
- Create: `evaluation/dataset_preprocessor.py`
- Create: `evaluation/ragas_eval.py`
- Create: `evaluation/llm_judge.py`
- Create: `evaluation/eval_runner.py`
- Create: `evaluation/report_writer.py`
- Create: `scripts/prepare_phase2_dataset.py`
- Create: `scripts/run_phase2_eval.py`
- Create: `data/eval/README.md`
- Create: `data/eval/phase2_eval_dataset.jsonl`
- Create: `tests/test_eval_dataset_loader.py`
- Create: `tests/test_eval_metrics.py`
- Create: `tests/test_eval_report_writer.py`
- Modify: `requirements.txt`
- Modify: `.env.example`
- Modify: `README.md`

### Task 1: Evaluation Scaffolding and Config Surface

**Files:**
- Create: `evaluation/__init__.py`
- Modify: `requirements.txt`
- Modify: `.env.example`
- Test: `tests/test_eval_dataset_loader.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_dataset_loader.py
def test_eval_module_importable():
    import evaluation  # noqa: F401
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_eval_dataset_loader.py::test_eval_module_importable -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluation'`

- [ ] **Step 3: Write minimal implementation**

```python
# evaluation/__init__.py
"""Phase 2 evaluation package."""
```

Update dependencies for evaluation and reports:

```txt
# requirements.txt additions
ragas
pandas
```

Add env knobs:

```env
# .env.example additions
EVAL_DATASET_PATH=data/eval/phase2_eval_dataset.jsonl
EVAL_OUTPUT_DIR=data/eval/reports
EVAL_MODEL=gemini-2.0-flash-001
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_eval_dataset_loader.py::test_eval_module_importable -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add evaluation/__init__.py requirements.txt .env.example tests/test_eval_dataset_loader.py
git commit -m "chore: scaffold phase 2 evaluation package and config"
```

### Task 2: Dataset Loader and Validation

**Files:**
- Create: `evaluation/dataset_loader.py`
- Create: `evaluation/dataset_preprocessor.py`
- Create: `data/eval/README.md`
- Create: `data/eval/phase2_eval_dataset.jsonl`
- Create: `scripts/prepare_phase2_dataset.py`
- Modify: `tests/test_eval_dataset_loader.py`
- Test: `tests/test_eval_dataset_loader.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_dataset_loader.py
from evaluation.dataset_loader import load_eval_dataset

def test_load_eval_dataset_returns_records(tmp_path):
    p = tmp_path / "dataset.jsonl"
    p.write_text('{"query":"Q1","expected_answer":"A1"}\n', encoding="utf-8")
    rows = load_eval_dataset(str(p))
    assert len(rows) == 1
    assert rows[0]["query"] == "Q1"
    assert rows[0]["expected_answer"] == "A1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_eval_dataset_loader.py::test_load_eval_dataset_returns_records -v`  
Expected: FAIL with missing loader

- [ ] **Step 3: Write minimal implementation**

```python
# evaluation/dataset_loader.py
import json

REQUIRED_KEYS = {"query", "expected_answer"}

def load_eval_dataset(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing = REQUIRED_KEYS - set(row)
            if missing:
                raise ValueError(f"Missing required keys: {sorted(missing)}")
            rows.append(row)
    return rows
```

Create dataset schema documentation and preprocess the provided PDF into JSONL.

Preprocessor responsibilities:

- Read `/Users/ssingodia/Desktop/RAG/RAG_evaluation_dataset - convertcsv (2).pdf`
- Extract table rows (Camelot first, Tabula fallback if needed)
- Map source columns to `query`, `expected_answer`, optional `context`
- Write `data/eval/phase2_eval_dataset.jsonl`
- Validate rows using `load_eval_dataset`

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_eval_dataset_loader.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add evaluation/dataset_loader.py evaluation/dataset_preprocessor.py scripts/prepare_phase2_dataset.py data/eval/README.md data/eval/phase2_eval_dataset.jsonl tests/test_eval_dataset_loader.py
git commit -m "feat: add phase 2 dataset loader and pdf-to-jsonl preprocessing for eval"
```

### Task 3: Retrieval/Answer Capture for Evaluation

**Files:**
- Create: `evaluation/eval_runner.py`
- Modify: `tests/test_eval_metrics.py`
- Test: `tests/test_eval_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_metrics.py
from evaluation.eval_runner import build_eval_sample

def test_build_eval_sample_shape():
    sample = build_eval_sample(
        query="What changed?",
        expected="Revenue increased",
        answer="Revenue increased in FY24.",
        context=[{"text":"Revenue increased in FY24.","page":2}],
    )
    assert sample["query"] == "What changed?"
    assert "retrieved_context" in sample
    assert sample["model_answer"].startswith("Revenue")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_eval_metrics.py::test_build_eval_sample_shape -v`  
Expected: FAIL with missing module/function

- [ ] **Step 3: Write minimal implementation**

```python
# evaluation/eval_runner.py
def build_eval_sample(query: str, expected: str, answer: str, context: list[dict]) -> dict:
    return {
        "query": query,
        "expected_answer": expected,
        "retrieved_context": context,
        "model_answer": answer,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_eval_metrics.py::test_build_eval_sample_shape -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add evaluation/eval_runner.py tests/test_eval_metrics.py
git commit -m "feat: add core evaluation sample builder"
```

### Task 4: RAGAS Metric Adapter

**Files:**
- Create: `evaluation/ragas_eval.py`
- Modify: `tests/test_eval_metrics.py`
- Test: `tests/test_eval_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_metrics.py
from evaluation.ragas_eval import compute_basic_ragas_like_metrics

def test_basic_ragas_like_metrics_keys():
    m = compute_basic_ragas_like_metrics(
        query="q",
        expected_answer="revenue increased",
        model_answer="revenue increased in fy24",
        retrieved_context=[{"text":"revenue increased in fy24"}],
    )
    assert "answer_relevance" in m
    assert "faithfulness" in m
    assert "context_relevance" in m
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_eval_metrics.py::test_basic_ragas_like_metrics_keys -v`  
Expected: FAIL with missing module/function

- [ ] **Step 3: Write minimal implementation**

```python
# evaluation/ragas_eval.py
def _token_overlap(a: str, b: str) -> float:
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(len(sa), 1)

def compute_basic_ragas_like_metrics(
    query: str,
    expected_answer: str,
    model_answer: str,
    retrieved_context: list[dict],
) -> dict:
    context_text = " ".join([c.get("text", "") for c in retrieved_context])
    return {
        "answer_relevance": _token_overlap(expected_answer, model_answer),
        "faithfulness": _token_overlap(model_answer, context_text),
        "context_relevance": _token_overlap(query, context_text),
    }
```

Note: this is a deterministic local fallback metric shim. In a follow-up step, wire true RAGAS if available in runtime.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_eval_metrics.py::test_basic_ragas_like_metrics_keys -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add evaluation/ragas_eval.py tests/test_eval_metrics.py
git commit -m "feat: add ragas-like local metric adapter for reproducible baseline"
```

### Task 5: LLM-as-Judge Scoring (Vertex)

**Files:**
- Create: `evaluation/llm_judge.py`
- Modify: `tests/test_eval_metrics.py`
- Test: `tests/test_eval_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_metrics.py
from evaluation.llm_judge import build_judge_prompt

def test_build_judge_prompt_contains_axes():
    p = build_judge_prompt("q", "gold", "pred")
    assert "correctness" in p.lower()
    assert "hallucination" in p.lower()
    assert "reasoning quality" in p.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_eval_metrics.py::test_build_judge_prompt_contains_axes -v`  
Expected: FAIL with missing module/function

- [ ] **Step 3: Write minimal implementation**

```python
# evaluation/llm_judge.py
import json
from generation.llm_pipeline import GeminiClient

def build_judge_prompt(query: str, expected: str, predicted: str) -> str:
    return f"""
You are an impartial evaluator.
Score:
1) correctness (1-5)
2) hallucination (yes/no)
3) reasoning quality (1-5)

Return strict JSON with keys:
correctness, hallucination, reasoning_quality, rationale

Query: {query}
Expected answer: {expected}
Predicted answer: {predicted}
""".strip()

def judge_with_llm(query: str, expected: str, predicted: str) -> dict:
    prompt = build_judge_prompt(query, expected, predicted)
    raw = GeminiClient().answer(prompt)
    try:
        return json.loads(raw)
    except Exception:
        return {
            "correctness": None,
            "hallucination": None,
            "reasoning_quality": None,
            "rationale": raw,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_eval_metrics.py::test_build_judge_prompt_contains_axes -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add evaluation/llm_judge.py tests/test_eval_metrics.py
git commit -m "feat: add vertex llm-judge prompt and parser"
```

### Task 6: Report Writer and Run Artifact Persistence

**Files:**
- Create: `evaluation/report_writer.py`
- Modify: `tests/test_eval_report_writer.py`
- Test: `tests/test_eval_report_writer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_report_writer.py
from evaluation.report_writer import summarize_results

def test_summarize_results_averages_metrics():
    rows = [
        {"ragas": {"answer_relevance": 1.0, "faithfulness": 0.5, "context_relevance": 0.5}},
        {"ragas": {"answer_relevance": 0.0, "faithfulness": 1.0, "context_relevance": 1.0}},
    ]
    s = summarize_results(rows)
    assert round(s["answer_relevance"], 2) == 0.50
    assert round(s["faithfulness"], 2) == 0.75
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_eval_report_writer.py::test_summarize_results_averages_metrics -v`  
Expected: FAIL with missing module/function

- [ ] **Step 3: Write minimal implementation**

```python
# evaluation/report_writer.py
import json
from datetime import datetime
from pathlib import Path

def summarize_results(rows: list[dict]) -> dict:
    keys = ["answer_relevance", "faithfulness", "context_relevance"]
    out = {}
    for k in keys:
        vals = [r["ragas"][k] for r in rows if "ragas" in r and k in r["ragas"]]
        out[k] = sum(vals) / len(vals) if vals else 0.0
    return out

def write_report(output_dir: str, rows: list[dict], summary: dict) -> str:
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    d = Path(output_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"phase2-eval-{ts}.json"
    payload = {"summary": summary, "rows": rows}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_eval_report_writer.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add evaluation/report_writer.py tests/test_eval_report_writer.py
git commit -m "feat: add phase 2 evaluation report summary and artifact writer"
```

### Task 7: CLI Runner Wiring

**Files:**
- Create: `scripts/run_phase2_eval.py`
- Modify: `evaluation/eval_runner.py`
- Modify: `README.md`
- Test: `tests/test_eval_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_metrics.py
from evaluation.eval_runner import evaluate_row

def test_evaluate_row_has_ragas_and_judge(monkeypatch):
    monkeypatch.setattr("evaluation.eval_runner.judge_with_llm", lambda *a, **k: {"correctness": 4})
    row = evaluate_row(
        {"query": "q", "expected_answer": "a"},
        answer="a",
        context=[{"text": "a"}],
    )
    assert "ragas" in row
    assert "judge" in row
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_eval_metrics.py::test_evaluate_row_has_ragas_and_judge -v`  
Expected: FAIL before wiring

- [ ] **Step 3: Write minimal implementation**

```python
# evaluation/eval_runner.py additions
from evaluation.llm_judge import judge_with_llm
from evaluation.ragas_eval import compute_basic_ragas_like_metrics

def evaluate_row(row: dict, answer: str, context: list[dict]) -> dict:
    ragas = compute_basic_ragas_like_metrics(
        query=row["query"],
        expected_answer=row["expected_answer"],
        model_answer=answer,
        retrieved_context=context,
    )
    judge = judge_with_llm(row["query"], row["expected_answer"], answer)
    out = dict(row)
    out["model_answer"] = answer
    out["retrieved_context"] = context
    out["ragas"] = ragas
    out["judge"] = judge
    return out
```

Implement runner script:

```python
# scripts/run_phase2_eval.py
# 1) load dataset
# 2) for each row, run existing rag query path (or placeholder adapter initially)
# 3) evaluate row
# 4) summarize + write report
# 5) print report path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_eval_metrics.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/run_phase2_eval.py evaluation/eval_runner.py README.md tests/test_eval_metrics.py
git commit -m "feat: wire phase 2 eval runner with ragas and llm-judge outputs"
```

### Task 8: End-to-End Evaluation Verification

**Files:**
- Modify: `README.md`
- Test: `tests/test_eval_dataset_loader.py`, `tests/test_eval_metrics.py`, `tests/test_eval_report_writer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_report_writer.py
from evaluation.report_writer import write_report

def test_write_report_creates_file(tmp_path):
    path = write_report(str(tmp_path), [{"ragas": {}}], {"answer_relevance": 0.0})
    assert path.endswith(".json")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_eval_report_writer.py::test_write_report_creates_file -v`  
Expected: FAIL if writer not complete

- [ ] **Step 3: Write minimal implementation**

Document commands in `README.md`:

```bash
pytest tests/test_eval_dataset_loader.py tests/test_eval_metrics.py tests/test_eval_report_writer.py -v
python scripts/run_phase2_eval.py
```

Expected:
- report JSON written to `data/eval/reports/`
- summary printed to terminal with average metric values

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_eval_dataset_loader.py tests/test_eval_metrics.py tests/test_eval_report_writer.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add README.md tests/test_eval_report_writer.py
git commit -m "docs: add phase 2 evaluation execution and verification instructions"
```

## Verification Checklist (Before Marking Phase 2 Complete)

- `pytest tests/test_eval_dataset_loader.py tests/test_eval_metrics.py tests/test_eval_report_writer.py -v` passes.
- `python scripts/run_phase2_eval.py` generates a report artifact in `data/eval/reports/`.
- Report includes per-row fields: `query`, `expected_answer`, `model_answer`, `retrieved_context`, `ragas`, `judge`.
- Summary includes averaged `answer_relevance`, `faithfulness`, `context_relevance`.
- LLM judge output is captured even when JSON parse fails (fallback rationale field).

## Notes / Guardrails

- If `ragas` install/runtime fails in environment, keep local deterministic metric adapter active and flag it clearly in report metadata.
- LLM judge should be best-effort: evaluation should not crash entire run when one judge call fails.
- Never commit real credential files or secrets. Keep `gcp-key.json` out of version control.
