from __future__ import annotations


def table_to_chunks(
    doc_id: str,
    page: int,
    table_id: int,
    headers: list[str],
    rows: list[list],
    max_rows_per_chunk: int,
) -> list[dict]:
    """Turn one extracted table into embeddable chunk dicts with table_json for UI."""
    chunks: list[dict] = []
    h = [str(x or "").strip() for x in headers]
    clean_rows: list[list[str]] = []
    for row in rows:
        clean_rows.append([str(c or "").strip() for c in row])

    if not clean_rows and not any(h):
        return []

    if not clean_rows:
        text = " | ".join(h)
        table_json = {"headers": h, "rows": []}
        cid = f"{doc_id}_p{page}_t{table_id}_c0"
        return [
            {
                "chunk_id": cid,
                "doc_id": doc_id,
                "page": page,
                "text": text.strip() or "(empty table)",
                "modality": "table",
                "table_id": table_id,
                "table_json": table_json,
            }
        ]

    for start in range(0, len(clean_rows), max(1, max_rows_per_chunk)):
        part_rows = clean_rows[start : start + max_rows_per_chunk]
        chunk_num = start // max(1, max_rows_per_chunk)
        lines = []
        for r in part_rows:
            pairs = []
            for i, cell in enumerate(r):
                col = h[i] if i < len(h) else f"col{i}"
                if cell:
                    pairs.append(f"{col}: {cell}")
            if pairs:
                lines.append("; ".join(pairs))
        text = "\n".join(lines) if lines else ""
        if not text.strip() and h:
            text = " | ".join(h)

        table_json = {"headers": h, "rows": part_rows}
        cid = f"{doc_id}_p{page}_t{table_id}_c{chunk_num}"
        chunks.append(
            {
                "chunk_id": cid,
                "doc_id": doc_id,
                "page": page,
                "text": text.strip() or "(empty table)",
                "modality": "table",
                "table_id": table_id,
                "table_json": table_json,
            }
        )
    return chunks
