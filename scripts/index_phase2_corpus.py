from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from configs.settings import Settings
from main import index_pdf


def main() -> None:
    settings = Settings()
    default_pdf = "/Users/ssingodia/Desktop/RAG/ifc-annual-report-2024-financials.pdf"
    pdf_path = os.getenv("PHASE2_CORPUS_PDF", default_pdf)
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Corpus PDF not found: {pdf_path}. Set PHASE2_CORPUS_PDF to a valid path."
        )
    index_pdf(str(path), settings)
    print(f"Indexed corpus PDF: {pdf_path}")
    print(f"FAISS index: {settings.faiss_index_path}.faiss")


if __name__ == "__main__":
    main()

