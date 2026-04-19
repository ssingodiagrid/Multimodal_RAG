from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.dataset_loader import load_eval_dataset
from evaluation.dataset_preprocessor import extract_eval_rows_from_pdf, write_jsonl


def main() -> None:
    input_pdf = os.getenv(
        "PHASE2_EVAL_SOURCE_PDF",
        "/Users/ssingodia/Desktop/RAG/RAG_evaluation_dataset - convertcsv (2).pdf",
    )
    output_jsonl = os.getenv("EVAL_DATASET_PATH", "data/eval/phase2_eval_dataset.jsonl")
    rows = extract_eval_rows_from_pdf(input_pdf)
    if not rows:
        raise ValueError(f"No evaluation rows extracted from {input_pdf}")
    path = write_jsonl(rows, output_jsonl)
    # Validation gate
    validated = load_eval_dataset(path)
    print(f"Prepared dataset: {path}")
    print(f"Rows: {len(validated)}")


if __name__ == "__main__":
    main()

