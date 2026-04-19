from ingestion.tables_normalize import table_to_chunks


def test_table_to_chunks_row_text():
    chunks = table_to_chunks(
        "doc",
        page=3,
        table_id=0,
        headers=["A", "B"],
        rows=[["1", "2"], ["3", "4"]],
        max_rows_per_chunk=10,
    )
    assert len(chunks) == 1
    assert chunks[0]["modality"] == "table"
    assert chunks[0]["page"] == 3
    assert "A: 1" in chunks[0]["text"]
    assert chunks[0]["table_json"]["headers"] == ["A", "B"]


def test_table_headers_only():
    chunks = table_to_chunks("d", 1, 0, ["X", "Y"], [], max_rows_per_chunk=5)
    assert len(chunks) == 1
    assert "X" in chunks[0]["text"]
