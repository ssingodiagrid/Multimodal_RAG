from generation.prompt_builder import build_grounded_prompt


def test_prompt_tags_table():
    p = build_grounded_prompt(
        "Q?",
        [
            {
                "text": "row facts",
                "page": 2,
                "chunk_id": "t1",
                "modality": "table",
            }
        ],
    )
    assert "[table p2]" in p
    assert "row facts" in p


def test_prompt_tags_image():
    p = build_grounded_prompt(
        "Q?",
        [
            {
                "text": "a chart showing growth",
                "page": 4,
                "chunk_id": "i1",
                "modality": "image",
                "asset_path": "data/x.png",
            }
        ],
    )
    assert "[figure p4]" in p
    assert "data/x.png" in p
    assert "figure shows" in p


def test_prompt_colpali_hint():
    p = build_grounded_prompt(
        "Q?",
        [{"text": "x", "page": 1, "chunk_id": "c", "modality": "text"}],
        include_colpali=True,
    )
    assert "ColPali" in p


def test_prompt_user_image_hint():
    p = build_grounded_prompt(
        "Does this match a figure in the report?",
        [{"text": "body", "page": 1, "chunk_id": "x", "modality": "text"}],
        include_user_image=True,
    )
    assert "attached a reference image" in p


def test_prompt_refinement_hint():
    p = build_grounded_prompt(
        "What is revenue?",
        [{"text": "body", "page": 1, "chunk_id": "x", "modality": "text"}],
        refined_query="Q4 revenue breakdown",
    )
    assert "Rewritten search query" in p
    assert "Q4 revenue breakdown" in p
    assert "Question: What is revenue?" in p
