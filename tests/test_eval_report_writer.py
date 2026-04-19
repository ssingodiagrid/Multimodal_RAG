from evaluation.report_writer import summarize_results, write_report


def test_summarize_results_averages_metrics():
    rows = [
        {
            "ragas": {
                "answer_relevance": 1.0,
                "faithfulness": 0.5,
                "context_relevance": 0.5,
            }
        },
        {
            "ragas": {
                "answer_relevance": 0.0,
                "faithfulness": 1.0,
                "context_relevance": 1.0,
            }
        },
    ]
    s = summarize_results(rows)
    assert round(s["answer_relevance"], 2) == 0.50
    assert round(s["faithfulness"], 2) == 0.75


def test_write_report_creates_file(tmp_path):
    path = write_report(str(tmp_path), [{"ragas": {}}], {"answer_relevance": 0.0})
    assert path.endswith(".json")

