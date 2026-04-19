from unittest.mock import MagicMock

from configs.settings import Settings
from retrieval.multihop import merge_contexts, parse_sub_query_json, should_multihop


def test_merge_dedupes_chunk_id():
    a = [
        {
            "chunk_id": "1",
            "text": "x",
            "page": 1,
            "score": 0.9,
            "score_source": "dense",
        }
    ]
    b = [
        {
            "chunk_id": "1",
            "text": "x",
            "page": 1,
            "score": 0.5,
            "score_source": "dense",
        },
        {
            "chunk_id": "2",
            "text": "y",
            "page": 2,
            "score": 0.8,
            "score_source": "dense",
        },
    ]
    m = merge_contexts(a, b, max_chunks=3)
    assert len(m) == 2
    assert m[0]["chunk_id"] == "1"


def test_should_multihop_heuristic():
    s = MagicMock(spec=Settings)
    s.enable_multi_hop = True
    s.multi_hop_mode = "heuristic"
    assert should_multihop("What is revenue?", s) is False
    assert should_multihop("Compare revenue and costs for FY24", s) is True


def test_should_multihop_always():
    s = MagicMock(spec=Settings)
    s.enable_multi_hop = True
    s.multi_hop_mode = "always"
    assert should_multihop("What is revenue?", s) is True


def test_parse_sub_query_json_raw():
    assert parse_sub_query_json('{"sub_query": "foo bar"}') == "foo bar"


def test_parse_sub_query_json_fenced():
    text = 'Here:\n```json\n{"sub_query": "x y"}\n```'
    assert parse_sub_query_json(text) == "x y"
