from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from configs.settings import Settings
from main import load_index
from retrieval.bm25 import save_sparse_corpus


def main() -> None:
    settings = Settings()
    store = load_index(settings)
    corpus = store.all_metadata()
    path = save_sparse_corpus(corpus, settings.sparse_index_path)
    print(f"Sparse corpus written: {path}")
    print(f"Rows: {len(corpus)}")


if __name__ == "__main__":
    main()

