from __future__ import annotations

import logging
from pathlib import Path

import fitz

logger = logging.getLogger(__name__)


def _resize_image_bytes(data: bytes, max_side: int, mime: str) -> tuple[bytes, str]:
    if max_side <= 0:
        return data, mime
    try:
        from io import BytesIO

        from PIL import Image
    except ImportError:
        return data, mime
    try:
        im = Image.open(BytesIO(data))
        im = im.convert("RGB") if im.mode not in ("RGB", "L") else im
        w, h = im.size
        side = max(w, h)
        if side <= max_side:
            return data, mime
        scale = max_side / float(side)
        nw, nh = int(w * scale), int(h * scale)
        im = im.resize((nw, nh), Image.Resampling.LANCZOS)
        buf = BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue(), "image/png"
    except Exception as exc:
        logger.warning("image resize skipped: %s", exc)
        return data, mime


def extract_images_from_pdf(
    pdf_path: str,
    assets_dir: str,
    doc_id: str,
    max_side: int,
) -> list[dict]:
    """Write images under assets_dir/doc_id; return metadata dicts for captioning."""
    base = Path(assets_dir) / doc_id
    base.mkdir(parents=True, exist_ok=True)
    out: list[dict] = []
    doc = fitz.open(pdf_path)
    try:
        img_counter = 0
        for page_index in range(len(doc)):
            page = doc[page_index]
            page_num = page_index + 1
            for img_entry in page.get_images(full=True):
                xref = img_entry[0]
                try:
                    info = doc.extract_image(xref)
                except Exception:
                    continue
                data = info.get("image") or b""
                ext = info.get("ext") or "png"
                mime = {
                    "png": "image/png",
                    "jpg": "image/jpeg",
                    "jpeg": "image/jpeg",
                    "gif": "image/gif",
                }.get(str(ext).lower(), "image/png")
                data, mime = _resize_image_bytes(data, max_side, mime)
                fname = f"page_{page_num}_img_{img_counter}.{ext if ext != 'jpeg' else 'jpg'}"
                if mime == "image/png":
                    fname = f"page_{page_num}_img_{img_counter}.png"
                path = base / fname
                path.write_bytes(data)
                out.append(
                    {
                        "page": page_num,
                        "image_index": img_counter,
                        "path": path,
                        "mime": mime,
                    }
                )
                img_counter += 1
    finally:
        doc.close()
    return out
