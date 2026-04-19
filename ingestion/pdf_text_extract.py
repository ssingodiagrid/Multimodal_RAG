"""Robust PDF text extraction: primary PyMuPDF, fallbacks for broken font encodings."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import fitz


def _ctrl_ratio(text: str) -> float:
    if not text:
        return 0.0
    bad = sum(
        1
        for c in text
        if ord(c) < 32 and c not in "\n\r\t\f"
    )
    return bad / len(text)


def _looks_like_bad_encoding(text: str) -> bool:
    if not text or len(text) < 40:
        return False
    if text.count("\x03") > max(5, len(text) // 300):
        return True
    if "\ufffd" in text and text.count("\ufffd") / len(text) > 0.02:
        return True
    if _ctrl_ratio(text) > 0.025:
        return True
    return False


def _readable_ratio(text: str) -> float:
    if not text:
        return 0.0
    good = sum(1 for c in text if c.isalnum() or c.isspace() or c in ".,;:!?%()-–—'\"/")
    return good / len(text)


def _text_from_fitz_blocks(page: "fitz.Page") -> str:
    blocks = page.get_text("blocks") or []
    parts: list[str] = []
    for b in blocks:
        if len(b) < 5:
            continue
        if b[-1] != 0:  # block_type 0 = text
            continue
        t = (b[4] or "").strip()
        if t:
            parts.append(t)
    return "\n\n".join(parts)


def _text_pdfplumber_page(pdf_path: str, page_index_zero: int) -> str:
    try:
        import pdfplumber
    except ImportError:
        return ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_index_zero >= len(pdf.pages):
                return ""
            t = pdf.pages[page_index_zero].extract_text(
                layout=False,
                x_tolerance=3,
                y_tolerance=3,
            )
            return (t or "").strip()
    except Exception:
        return ""


def extract_page_text(page: "fitz.Page", pdf_path: str, page_index_zero: int) -> str:
    """Best-effort UTF-8 text for one page."""
    # 1) Default text (reading order)
    raw = page.get_text("text") or ""
    text = raw.strip()

    # 2) Sorted by position (sometimes fixes order / spacing)
    if _looks_like_bad_encoding(text):
        raw2 = page.get_text("text", sort=True) or ""
        if _readable_ratio(raw2) > _readable_ratio(text) or not _looks_like_bad_encoding(
            raw2
        ):
            text = raw2.strip()

    # 3) Block merge (different code path in MuPDF)
    if _looks_like_bad_encoding(text):
        block_t = _text_from_fitz_blocks(page)
        if block_t and (
            _readable_ratio(block_t) > _readable_ratio(text)
            or not _looks_like_bad_encoding(block_t)
        ):
            text = block_t

    # 4) pdfplumber (often uses a different text pipeline)
    if _looks_like_bad_encoding(text):
        pl_t = _text_pdfplumber_page(pdf_path, page_index_zero)
        if pl_t and (
            _readable_ratio(pl_t) > _readable_ratio(text)
            or not _looks_like_bad_encoding(pl_t)
        ):
            text = pl_t

    # Strip control chars that leak into chunks (keep newline for chunker)
    text = "".join(c for c in text if ord(c) >= 32 or c in "\n\r\t\f")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = _filter_garbage_lines(text)
    return text.strip()


def _filter_garbage_lines(text: str) -> str:
    """Drop lines that look like broken font mapping (low vowel ratio in long runs)."""
    out: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if _ctrl_ratio(s) > 0.02:
            continue
        alpha = "".join(c for c in s if c.isalpha())
        if len(alpha) > 30:
            vowels = sum(1 for c in alpha.lower() if c in "aeiouy")
            if vowels / len(alpha) < 0.12:
                continue
        out.append(s)
    return "\n".join(out)
