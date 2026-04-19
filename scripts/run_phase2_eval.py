from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from configs.settings import Settings
from evaluation.dataset_loader import load_eval_dataset
from evaluation.eval_runner import evaluate_row
from evaluation.report_writer import summarize_results, write_report
from main import load_index, run_rag_query
from retrieval.embedder import Embedder


def main() -> None:
    settings = Settings()
    dataset_path = os.getenv("EVAL_DATASET_PATH", "data/eval/phase2_eval_dataset.jsonl")
    output_dir = os.getenv("EVAL_OUTPUT_DIR", "data/eval/reports")

    rows = load_eval_dataset(dataset_path)
    store = load_index(settings)
    embedder = None
    embedder_error = None
    try:
        embedder = Embedder(settings.embedding_model)
    except Exception as exc:
        embedder_error = str(exc)

    evaluated = []
    for row in rows:
        query = row["query"]
        try:
            if embedder is None:
                context = []
                answer = f"RAG_RUNTIME_ERROR: embedder_unavailable ({embedder_error})"
            else:
                answer, context = run_rag_query(query, store, embedder, settings)
        except Exception as exc:
            answer = f"RAG_RUNTIME_ERROR: {exc}"
            context = []
        evaluated.append(evaluate_row(row, answer=answer, context=context))

    summary = summarize_results(evaluated)
    report_path = write_report(output_dir, evaluated, summary)
    print(f"Phase 2 report written: {report_path}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()

