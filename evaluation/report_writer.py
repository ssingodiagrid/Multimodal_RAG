import json
from datetime import datetime
from pathlib import Path


def summarize_results(rows: list[dict]) -> dict:
    keys = ["answer_relevance", "faithfulness", "context_relevance"]
    out = {}
    for k in keys:
        vals = [r["ragas"][k] for r in rows if "ragas" in r and k in r["ragas"]]
        out[k] = sum(vals) / len(vals) if vals else 0.0
    return out


def write_report(output_dir: str, rows: list[dict], summary: dict) -> str:
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    d = Path(output_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"phase2-eval-{ts}.json"
    payload = {"summary": summary, "rows": rows}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)

