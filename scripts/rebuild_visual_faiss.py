#!/usr/bin/env python3
"""Build Phase 6 visual FAISS from existing dense index metadata (no PDF re-parse).

Use when ENABLE_VISUAL_RETRIEVAL was off during the last index, or CLIP step failed.
Requires: image rows in {FAISS_INDEX_PATH}.meta.json and files at asset_path (repo root).

  python scripts/rebuild_visual_faiss.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# Before torch / huggingface: reduces segfault risk on macOS (OpenMP / tokenizers).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("rebuild_visual")

from configs.settings import Settings, ensure_phase6_fields
from retrieval.visual_index import build_visual_faiss_index


def main() -> int:
    settings = Settings()
    ensure_phase6_fields(settings)
    if not settings.enable_visual_retrieval:
        log.error("Set ENABLE_VISUAL_RETRIEVAL=true in .env first.")
        return 1
    meta_path = Path(str(settings.faiss_index_path) + ".meta.json")
    if not meta_path.is_file():
        log.error("Missing dense metadata: %s (index a PDF first)", meta_path)
        return 1
    chunks = json.loads(meta_path.read_text(encoding="utf-8"))
    n_img = sum(1 for c in chunks if c.get("modality") == "image" and c.get("asset_path"))
    log.info("Loaded %d rows (%d with image + asset_path)", len(chunks), n_img)
    if n_img == 0:
        log.error(
            "No image chunks in metadata. Enable ENABLE_IMAGE_CAPTIONS and re-index the PDF."
        )
        return 1
    out = build_visual_faiss_index(chunks, settings, _ROOT)
    if out is None:
        log.error(
            "Build returned None (no files on disk under asset_path, or CLIP/embed failed — see logs above)."
        )
        return 1
    log.info("OK: %s (%d vectors)", settings.visual_faiss_index_path, out.index.ntotal)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
