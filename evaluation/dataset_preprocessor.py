import json
from pathlib import Path

import fitz


def _pick_separator(header: str) -> str:
    if "|" in header:
        return "|"
    if "," in header:
        return ","
    if "\t" in header:
        return "\t"
    return ","


def _header_map(header_line: str, sep: str) -> dict[str, int]:
    cols = [c.strip().lower() for c in header_line.split(sep)]
    idx = {}
    for i, c in enumerate(cols):
        if c in {"question", "query", "user_query"}:
            idx["query"] = i
        elif c in {"answer", "expected_answer", "ground_truth", "gold_answer"}:
            idx["expected_answer"] = i
        elif c in {"context", "reference_context"}:
            idx["context"] = i
    return idx


def extract_eval_rows_from_pdf(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    lines: list[str] = []
    for page in doc:
        lines.extend([ln.strip() for ln in page.get_text("text").splitlines() if ln.strip()])
    doc.close()
    if not lines:
        return []

    header = lines[0]
    sep = _pick_separator(header)
    col_map = _header_map(header, sep)
    if "query" not in col_map or "expected_answer" not in col_map:
        # Fallback for unstructured extraction: pair alternating lines
        rows = []
        for i in range(0, len(lines) - 1, 2):
            rows.append({"query": lines[i], "expected_answer": lines[i + 1]})
        return rows

    rows: list[dict] = []
    for line in lines[1:]:
        parts = [p.strip() for p in line.split(sep)]
        if len(parts) <= max(col_map.values()):
            continue
        row = {
            "query": parts[col_map["query"]],
            "expected_answer": parts[col_map["expected_answer"]],
        }
        if "context" in col_map and len(parts) > col_map["context"]:
            row["context"] = parts[col_map["context"]]
        rows.append(row)
    return rows


def write_jsonl(rows: list[dict], output_path: str) -> str:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return str(out)

