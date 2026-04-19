from configs.settings import Settings


def test_phase4_cache_settings_exist():
    s = Settings()
    assert hasattr(s, "enable_semantic_cache")
    assert hasattr(s, "semantic_cache_threshold")
    assert hasattr(s, "semantic_cache_max_entries")
    assert hasattr(s, "semantic_cache_path")


def test_phase4_multihop_settings_exist():
    s = Settings()
    assert hasattr(s, "enable_multi_hop")
    assert hasattr(s, "multi_hop_mode")
    assert hasattr(s, "multi_hop_merged_top_k")


def test_multi_hop_mode_invalid_falls_back_to_heuristic(monkeypatch):
    monkeypatch.setenv("MULTI_HOP_MODE", "bogus")
    s = Settings()
    assert s.multi_hop_mode == "heuristic"
