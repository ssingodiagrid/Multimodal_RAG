# Phase 2 Evaluation Dataset

Primary source PDF:

- `/Users/ssingodia/Desktop/RAG/RAG_evaluation_dataset - convertcsv (2).pdf`

Preparation command:

```bash
python scripts/prepare_phase2_dataset.py
```

Output JSONL:

- `data/eval/phase2_eval_dataset.jsonl`

Required per-row keys:

- `query`
- `expected_answer`

Optional keys:

- `context`
- `metadata`

