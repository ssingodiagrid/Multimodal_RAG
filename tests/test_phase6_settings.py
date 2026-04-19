from configs.settings import Settings, ensure_phase6_fields
from retrieval.visual_index import should_run_visual_merge, visual_merge_gate_intent


def test_phase6_flags_exist():
    s = Settings()
    assert hasattr(s, "enable_visual_retrieval")
    assert hasattr(s, "visual_faiss_index_path")
    assert hasattr(s, "visual_embedding_model")
    assert hasattr(s, "visual_top_k")
    assert hasattr(s, "visual_fusion_lambda")
    assert hasattr(s, "visual_device")
    assert hasattr(s, "visual_batch_size")
    assert hasattr(s, "visual_for_image_intent_only")


def test_ensure_phase6_backfill():
    class Old:
        pass

    o = Old()
    ensure_phase6_fields(o)
    assert getattr(o, "enable_visual_retrieval") is not None


def test_visual_merge_gate_heuristic_when_router_intent_missing():
    s = Settings()
    s.enable_visual_retrieval = True
    s.visual_for_image_intent_only = True
    gate = visual_merge_gate_intent(s, None, "show the image from the figure")
    assert gate == "image"
    assert should_run_visual_merge(s, gate) is True


def test_visual_merge_gate_respects_explicit_intent():
    s = Settings()
    s.enable_visual_retrieval = True
    s.visual_for_image_intent_only = True
    assert visual_merge_gate_intent(s, "table", "show the image") == "table"
    assert should_run_visual_merge(s, "table") is False


def test_visual_merge_gate_when_image_intent_only_off():
    s = Settings()
    s.enable_visual_retrieval = True
    s.visual_for_image_intent_only = False
    assert visual_merge_gate_intent(s, None, "anything") is None
    assert should_run_visual_merge(s, None) is True


def test_colpali_settings_exist():
    from configs.settings import ensure_colpali_fields

    s = Settings()
    ensure_colpali_fields(s)
    assert hasattr(s, "enable_colpali_index")
    assert hasattr(s, "enable_colpali_retrieval")
    assert hasattr(s, "colpali_model_id")
