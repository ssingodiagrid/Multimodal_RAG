"""Rasterize PDF regions (figure crops or full pages) for Phase 5 image indexing."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path

from ingestion.images_extract import _resize_image_bytes

logger = logging.getLogger(__name__)

# Caption line in financial / report PDFs (often below the graphic).
FIGURE_RE = re.compile(
    r"\b(?:Figure|Fig\.)\s+\d+[a-z]?\s*[:\.]?", re.IGNORECASE
)


def _figure_sort_key(label: str) -> str:
    m = re.search(r"(?:Figure|Fig\.)\s+(\d+)", label, re.IGNORECASE)
    return m.group(1).zfill(4) if m else label


def _line_union_rect(line: dict) -> "fitz.Rect | None":
    import fitz

    rects = [fitz.Rect(s["bbox"]) for s in line.get("spans", []) if "bbox" in s]
    if not rects:
        return None
    u = rects[0]
    for r in rects[1:]:
        u |= r
    return u


def _caption_lines_union_rect(lines: list[dict], start_i: int) -> tuple[str, "fitz.Rect"]:
    """
    Full PDF caption: the line containing 'Figure N' plus tightly following lines
    in the same block (multi-line titles), merged for bbox union.
    """
    line = lines[start_i]
    union = _line_union_rect(line)
    if union is None:
        return "", fitz.Rect(0, 0, 0, 0)
    line_text = "".join(s.get("text") or "" for s in line.get("spans", []))
    parts = [line_text.strip()]
    prev_y1 = float(line["bbox"][3])
    j = start_i + 1
    while j < len(lines):
        nl = lines[j]
        nt = "".join(s.get("text") or "" for s in nl.get("spans", [])).strip()
        if not nt:
            break
        if re.match(r"^\s*(?:Figure|Fig\.)\s*\d+", nt, re.IGNORECASE):
            break
        y0 = float(nl["bbox"][1])
        if y0 - prev_y1 > 22.0:
            break
        parts.append(nt)
        nu = _line_union_rect(nl)
        if nu is not None:
            union |= nu
        prev_y1 = float(nl["bbox"][3])
        j += 1
        if sum(len(x) for x in parts) > 520:
            break
    caption = " ".join(parts)
    caption = re.sub(r"\s+", " ", caption).strip()
    return caption, union


def _group_figure_regions(page: "fitz.Page") -> list[tuple[str, "fitz.Rect", str]]:
    """One entry per distinct figure number: label, caption bbox union, PDF caption text."""
    per_num: dict[str, list[fitz.Rect]] = defaultdict(list)
    per_caption: dict[str, str] = {}
    d = page.get_text("dict") or {}
    for block in d.get("blocks", []):
        if block.get("type") != 0:
            continue
        lines = block.get("lines", [])
        for i, line in enumerate(lines):
            spans = line.get("spans", [])
            if not spans:
                continue
            line_text = "".join(s.get("text") or "" for s in spans)
            if not FIGURE_RE.search(line_text):
                continue
            m = FIGURE_RE.search(line_text)
            raw_lbl = (m.group(0) if m else "Figure").strip()
            num_m = re.search(r"(\d+)", raw_lbl)
            key = num_m.group(1) if num_m else raw_lbl
            caption, union = _caption_lines_union_rect(lines, i)
            if union.width < 1 or union.height < 1:
                continue
            per_num[key].append(union)
            prev = per_caption.get(key, "")
            c = caption.strip()
            if len(c) > len(prev):
                per_caption[key] = c

    out: list[tuple[str, fitz.Rect, str]] = []
    for num, rects in sorted(per_num.items(), key=lambda kv: _figure_sort_key(kv[0])):
        u = rects[0]
        for r in rects[1:]:
            u |= r
        cap = per_caption.get(num, "").strip() or f"Figure {num}"
        out.append((f"Figure {num}", u, cap))
    return out


def _expand_clip_for_figure(page: "fitz.Page", caption_rect: "fitz.Rect") -> "fitz.Rect":
    """Expand mostly upward from caption so the diagram above is included."""
    import fitz

    pr = page.rect
    up = 520.0
    down = 80.0
    clip = fitz.Rect(
        pr.x0,
        max(pr.y0, caption_rect.y0 - up),
        pr.x1,
        min(pr.y1, caption_rect.y1 + down),
    )
    clip &= pr
    if clip.width < 72.0 or clip.height < 72.0:
        return pr
    return clip


def extract_page_renders_from_pdf(
    pdf_path: str,
    assets_dir: str,
    doc_id: str,
    dpi: float,
    max_side: int,
    strategy: str = "figures",
) -> list[dict]:
    """
    Write PNGs under assets_dir/doc_id from page pixmaps.

    strategy:
      - figures: crop around each "Figure N" caption (vector-friendly).
      - full_pages: one raster per page (large; use for scans or when figures lack labels).
    """
    import fitz

    strat = (strategy or "figures").lower()
    if strat not in ("figures", "full_pages"):
        strat = "figures"

    base = Path(assets_dir) / doc_id
    base.mkdir(parents=True, exist_ok=True)
    out: list[dict] = []
    render_counter = 0
    doc = fitz.open(pdf_path)
    try:
        zoom = max(36.0, min(float(dpi), 300.0)) / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for page_index in range(len(doc)):
            page = doc[page_index]
            page_num = page_index + 1
            jobs: list[tuple[str, fitz.Rect, str]] = []
            if strat == "full_pages":
                jobs.append(("", page.rect, ""))
            else:
                for label, union_r, pdf_caption in _group_figure_regions(page):
                    jobs.append(
                        (
                            label,
                            _expand_clip_for_figure(page, union_r),
                            pdf_caption,
                        )
                    )

            for label, clip, pdf_caption in jobs:
                try:
                    pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
                    png_bytes = pix.tobytes("png")
                except Exception as exc:
                    logger.warning(
                        "page pixmap failed doc=%s page=%s: %s", doc_id, page_num, exc
                    )
                    continue
                data, mime = _resize_image_bytes(png_bytes, max_side, "image/png")
                fname = f"page_{page_num}_render_{render_counter}.png"
                path = base / fname
                path.write_bytes(data)
                out.append(
                    {
                        "page": page_num,
                        "image_index": render_counter,
                        "path": path,
                        "mime": mime,
                        "label": label or None,
                        "pdf_caption": pdf_caption.strip() or None,
                    }
                )
                render_counter += 1
    finally:
        doc.close()

    if strat == "figures" and not out:
        logger.info(
            "No figure-caption regions found for pixmap indexing (doc=%s). "
            "Try IMAGE_PAGE_RENDER_STRATEGY=full_pages if you need every page rasterized.",
            doc_id,
        )
    return out
