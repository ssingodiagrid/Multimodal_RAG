import json

REQUIRED_KEYS = {"query", "expected_answer"}


def load_eval_dataset(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing = REQUIRED_KEYS - set(row)
            if missing:
                raise ValueError(f"Missing required keys: {sorted(missing)}")
            rows.append(row)
    return rows

