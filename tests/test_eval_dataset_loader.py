from evaluation.dataset_loader import load_eval_dataset


def test_eval_module_importable():
    import evaluation  # noqa: F401


def test_load_eval_dataset_returns_records(tmp_path):
    p = tmp_path / "dataset.jsonl"
    p.write_text('{"query":"Q1","expected_answer":"A1"}\n', encoding="utf-8")
    rows = load_eval_dataset(str(p))
    assert len(rows) == 1
    assert rows[0]["query"] == "Q1"
    assert rows[0]["expected_answer"] == "A1"

