from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _tables_pdfplumber(pdf_path: str) -> list[dict]:
    try:
        import pdfplumber
    except ImportError:
        return []
    out: list[dict] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            try:
                tables = page.extract_tables() or []
            except Exception:
                continue
            for t_idx, table in enumerate(tables):
                if not table or len(table) < 1:
                    continue
                headers = [str(c or "") for c in table[0]]
                rows = table[1:] if len(table) > 1 else []
                out.append(
                    {
                        "page": page_idx,
                        "table_index": t_idx,
                        "headers": headers,
                        "rows": rows,
                    }
                )
    return out


def _tables_camelot(pdf_path: str) -> list[dict]:
    try:
        import camelot
    except ImportError:
        return []
    out: list[dict] = []
    try:
        tables = camelot.read_pdf(pdf_path, pages="all", flavor="stream")
    except Exception as exc:
        logger.warning("camelot read failed: %s", exc)
        return []
    for t_idx, t in enumerate(tables):
        try:
            df = t.df
            headers = [str(x) for x in df.iloc[0].tolist()]
            rows = df.iloc[1:].values.tolist()
            page = int(t.page)
        except Exception:
            continue
        out.append(
            {
                "page": page,
                "table_index": t_idx,
                "headers": headers,
                "rows": rows,
            }
        )
    return out


def extract_tables_from_pdf(pdf_path: str, prefer_camelot: bool = False) -> list[dict]:
    """Return list of {page, table_index, headers, rows}. Empty if no library works."""
    if prefer_camelot:
        got = _tables_camelot(pdf_path)
        if got:
            return got
    got = _tables_pdfplumber(pdf_path)
    if got:
        return got
    if not prefer_camelot:
        got = _tables_camelot(pdf_path)
        if got:
            return got
    return []
