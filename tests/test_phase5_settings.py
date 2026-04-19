from configs.settings import Settings


def test_phase5_flags_exist():
    s = Settings()
    assert hasattr(s, "enable_table_extraction")
    assert hasattr(s, "enable_image_captions")
    assert hasattr(s, "enable_image_page_renders")
    assert hasattr(s, "image_page_render_dpi")
    assert hasattr(s, "image_page_render_strategy")
    assert hasattr(s, "enable_modality_router")
    assert hasattr(s, "assets_dir")
    assert hasattr(s, "table_max_rows_per_chunk")
