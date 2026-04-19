"""Full-page rasterization for ColPali-style document indexing (layout + visuals)."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def rasterize_pdf_pages_to_dir(
    pdf_path: str,
    out_dir: Path,
    *,
    dpi: float = 150.0,
    max_pages: int = 0,
) -> list[tuple[int, Path]]:
    """
    Write one PNG per PDF page under out_dir (filenames colpali_page_{n}.png).

    Returns list of (1-based page number, absolute path).
    """
    import fitz

    out_dir.mkdir(parents=True, exist_ok=True)
    zoom = max(72.0, min(float(dpi), 220.0)) / 72.0
    mat = fitz.Matrix(zoom, zoom)
    doc = fitz.open(pdf_path)
    out: list[tuple[int, Path]] = []
    try:
        n = len(doc)
        limit = n if max_pages <= 0 else min(n, max_pages)
        for i in range(limit):
            page = doc[i]
            page_num = i + 1
            try:
                pix = page.get_pixmap(matrix=mat, alpha=False)
            except Exception as exc:
                logger.warning("ColPali raster skip page %s: %s", page_num, exc)
                continue
            path = out_dir / f"colpali_page_{page_num}.png"
            path.write_bytes(pix.tobytes("png"))
            out.append((page_num, path.resolve()))
    finally:
        doc.close()
    return out
