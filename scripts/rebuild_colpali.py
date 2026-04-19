#!/usr/bin/env python3
"""Build ColPali page index for a PDF on disk (no full text re-embed)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

from configs.settings import Settings, ensure_colpali_fields
from retrieval.colpali_retrieval import build_colpali_page_index


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path, help="Path to PDF")
    args = ap.parse_args()
    p = args.pdf.expanduser().resolve()
    if not p.is_file():
        print("PDF not found:", p, file=sys.stderr)
        return 1
    s = Settings()
    ensure_colpali_fields(s)
    if not s.enable_colpali_index:
        print("Set ENABLE_COLPALI_INDEX=true in .env", file=sys.stderr)
        return 1
    doc_id = p.stem
    raw_cd = getattr(s, "colpali_device", "") or ""
    cdev = raw_cd.strip() or None
    out = build_colpali_page_index(
        str(p),
        doc_id,
        repo_root=_ROOT,
        assets_dir=s.assets_dir,
        index_dir=s.colpali_index_dir,
        model_id=s.colpali_model_id,
        page_dpi=float(getattr(s, "colpali_page_dpi", 150.0)),
        max_pages=int(getattr(s, "colpali_max_index_pages", 0)),
        device=cdev,
    )
    if out is None:
        return 1
    print("OK:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
