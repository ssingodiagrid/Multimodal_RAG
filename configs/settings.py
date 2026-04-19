from dataclasses import dataclass, field
import os


def _read_multi_hop_mode() -> str:
    m = os.getenv("MULTI_HOP_MODE", "heuristic").lower()
    return m if m in ("off", "heuristic", "always") else "heuristic"


def _read_image_page_render_strategy() -> str:
    s = os.getenv("IMAGE_PAGE_RENDER_STRATEGY", "figures").lower()
    return s if s in ("figures", "full_pages") else "figures"


@dataclass
class Settings:
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    faiss_index_path: str = os.getenv("FAISS_INDEX_PATH", "data/parsed/faiss.index")
    # Vertex AI model id (not the AI Studio model name)
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-001")
    gcp_project_id: str = os.getenv("GCP_PROJECT_ID", "")
    gcp_location: str = os.getenv("GCP_LOCATION", "us-central1")
    eval_dataset_path: str = os.getenv(
        "EVAL_DATASET_PATH", "data/eval/phase2_eval_dataset.jsonl"
    )
    eval_output_dir: str = os.getenv("EVAL_OUTPUT_DIR", "data/eval/reports")
    hybrid_alpha: float = float(os.getenv("HYBRID_ALPHA", "0.7"))
    dense_top_k: int = int(os.getenv("DENSE_TOP_K", "20"))
    sparse_top_k: int = int(os.getenv("SPARSE_TOP_K", "20"))
    hybrid_top_n: int = int(os.getenv("HYBRID_TOP_N", "20"))
    rerank_top_k: int = int(os.getenv("RERANK_TOP_K", "5"))
    enable_hybrid: bool = os.getenv("ENABLE_HYBRID", "true").lower() == "true"
    enable_rerank: bool = os.getenv("ENABLE_RERANK", "true").lower() == "true"
    rerank_model: str = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    sparse_index_path: str = os.getenv("SPARSE_INDEX_PATH", "data/parsed/bm25_corpus.json")
    enable_semantic_cache: bool = (
        os.getenv("ENABLE_SEMANTIC_CACHE", "false").lower() == "true"
    )
    semantic_cache_threshold: float = float(
        os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.92")
    )
    semantic_cache_max_entries: int = int(
        os.getenv("SEMANTIC_CACHE_MAX_ENTRIES", "500")
    )
    semantic_cache_path: str = os.getenv(
        "SEMANTIC_CACHE_PATH", "data/cache/semantic_cache"
    )
    enable_multi_hop: bool = os.getenv("ENABLE_MULTI_HOP", "false").lower() == "true"
    multi_hop_mode: str = field(default_factory=_read_multi_hop_mode)
    multi_hop_merged_top_k: int = int(
        os.getenv("MULTI_HOP_MERGED_TOP_K", os.getenv("TOP_K", "5"))
    )
    # Optional: LLM rewrite before embedding; merge raw + refined retrieval hits
    enable_query_refinement: bool = (
        os.getenv("ENABLE_QUERY_REFINEMENT", "false").lower() == "true"
    )
    enable_table_extraction: bool = (
        os.getenv("ENABLE_TABLE_EXTRACTION", "false").lower() == "true"
    )
    enable_image_captions: bool = (
        os.getenv("ENABLE_IMAGE_CAPTIONS", "true").lower() == "true"
    )
    enable_modality_router: bool = (
        os.getenv("ENABLE_MODALITY_ROUTER", "true").lower() == "true"
    )
    router_use_llm: bool = os.getenv("ROUTER_USE_LLM", "false").lower() == "true"
    table_max_rows_per_chunk: int = int(
        os.getenv("TABLE_MAX_ROWS_PER_CHUNK", "20")
    )
    image_caption_max_side: int = int(
        os.getenv("IMAGE_CAPTION_MAX_SIDE", "1024")
    )
    enable_image_page_renders: bool = (
        os.getenv("ENABLE_IMAGE_PAGE_RENDERS", "true").lower() == "true"
    )
    image_page_render_dpi: float = float(os.getenv("IMAGE_PAGE_RENDER_DPI", "120"))
    image_page_render_strategy: str = field(default_factory=_read_image_page_render_strategy)
    assets_dir: str = os.getenv("ASSETS_DIR", "data/parsed/assets")
    table_extractor_prefer_camelot: bool = (
        os.getenv("TABLE_EXTRACTOR", "pdfplumber").lower() == "camelot"
    )
    # Phase 6: CLIP-class visual index (separate FAISS from text)
    enable_visual_retrieval: bool = (
        os.getenv("ENABLE_VISUAL_RETRIEVAL", "true").lower() == "true"
    )
    visual_faiss_index_path: str = os.getenv(
        "VISUAL_FAISS_INDEX_PATH", "data/parsed/faiss.index.visual"
    )
    visual_embedding_model: str = os.getenv(
        "VISUAL_EMBEDDING_MODEL", "clip-ViT-B-32"
    )
    visual_top_k: int = int(os.getenv("VISUAL_TOP_K", "20"))
    visual_fusion_lambda: float = float(os.getenv("VISUAL_FUSION_LAMBDA", "0.65"))
    visual_device: str = os.getenv("VISUAL_DEVICE", "")
    visual_batch_size: int = int(os.getenv("VISUAL_BATCH_SIZE", "8"))
    visual_for_image_intent_only: bool = (
        os.getenv("VISUAL_FOR_IMAGE_INTENT_ONLY", "true").lower() == "true"
    )
    # ColPali-style late interaction over full-page rasters (separate from CLIP index)
    enable_colpali_index: bool = (
        os.getenv("ENABLE_COLPALI_INDEX", "false").lower() == "true"
    )
    enable_colpali_retrieval: bool = (
        os.getenv("ENABLE_COLPALI_RETRIEVAL", "false").lower() == "true"
    )
    colpali_model_id: str = os.getenv(
        "COLPALI_MODEL_ID", "vidore/colpali-v1.2"
    )
    colpali_index_dir: str = os.getenv("COLPALI_INDEX_DIR", "data/parsed/colpali")
    colpali_page_dpi: float = float(os.getenv("COLPALI_PAGE_DPI", "150"))
    colpali_max_index_pages: int = int(os.getenv("COLPALI_MAX_INDEX_PAGES", "0"))
    colpali_top_k: int = int(os.getenv("COLPALI_TOP_K", "3"))
    colpali_device: str = os.getenv("COLPALI_DEVICE", "")
    colpali_max_images_for_llm: int = int(
        os.getenv("COLPALI_MAX_IMAGES_FOR_LLM", "4")
    )


def ensure_phase4_fields(settings: Settings) -> None:
    """Backfill Phase 4 attributes if missing (e.g. Streamlit hot-reload of an older Settings class)."""
    if not hasattr(settings, "enable_semantic_cache"):
        settings.enable_semantic_cache = (
            os.getenv("ENABLE_SEMANTIC_CACHE", "false").lower() == "true"
        )
    if not hasattr(settings, "semantic_cache_threshold"):
        settings.semantic_cache_threshold = float(
            os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.92")
        )
    if not hasattr(settings, "semantic_cache_max_entries"):
        settings.semantic_cache_max_entries = int(
            os.getenv("SEMANTIC_CACHE_MAX_ENTRIES", "500")
        )
    if not hasattr(settings, "semantic_cache_path"):
        settings.semantic_cache_path = os.getenv(
            "SEMANTIC_CACHE_PATH", "data/cache/semantic_cache"
        )
    if not hasattr(settings, "enable_multi_hop"):
        settings.enable_multi_hop = (
            os.getenv("ENABLE_MULTI_HOP", "false").lower() == "true"
        )
    if not hasattr(settings, "multi_hop_mode"):
        settings.multi_hop_mode = _read_multi_hop_mode()
    if not hasattr(settings, "multi_hop_merged_top_k"):
        settings.multi_hop_merged_top_k = int(
            os.getenv("MULTI_HOP_MERGED_TOP_K", os.getenv("TOP_K", "5"))
        )
    if not hasattr(settings, "enable_query_refinement"):
        settings.enable_query_refinement = (
            os.getenv("ENABLE_QUERY_REFINEMENT", "false").lower() == "true"
        )


def ensure_phase5_fields(settings: Settings) -> None:
    """Backfill Phase 5 attributes if missing (e.g. Streamlit hot-reload)."""
    if not hasattr(settings, "enable_table_extraction"):
        settings.enable_table_extraction = (
            os.getenv("ENABLE_TABLE_EXTRACTION", "false").lower() == "true"
        )
    if not hasattr(settings, "enable_image_captions"):
        settings.enable_image_captions = (
            os.getenv("ENABLE_IMAGE_CAPTIONS", "true").lower() == "true"
        )
    if not hasattr(settings, "enable_modality_router"):
        settings.enable_modality_router = (
            os.getenv("ENABLE_MODALITY_ROUTER", "true").lower() == "true"
        )
    if not hasattr(settings, "router_use_llm"):
        settings.router_use_llm = (
            os.getenv("ROUTER_USE_LLM", "false").lower() == "true"
        )
    if not hasattr(settings, "table_max_rows_per_chunk"):
        settings.table_max_rows_per_chunk = int(
            os.getenv("TABLE_MAX_ROWS_PER_CHUNK", "20")
        )
    if not hasattr(settings, "image_caption_max_side"):
        settings.image_caption_max_side = int(
            os.getenv("IMAGE_CAPTION_MAX_SIDE", "1024")
        )
    if not hasattr(settings, "enable_image_page_renders"):
        settings.enable_image_page_renders = (
            os.getenv("ENABLE_IMAGE_PAGE_RENDERS", "true").lower() == "true"
        )
    if not hasattr(settings, "image_page_render_dpi"):
        settings.image_page_render_dpi = float(
            os.getenv("IMAGE_PAGE_RENDER_DPI", "120")
        )
    if not hasattr(settings, "image_page_render_strategy"):
        settings.image_page_render_strategy = _read_image_page_render_strategy()
    if not hasattr(settings, "assets_dir"):
        settings.assets_dir = os.getenv("ASSETS_DIR", "data/parsed/assets")
    if not hasattr(settings, "table_extractor_prefer_camelot"):
        settings.table_extractor_prefer_camelot = (
            os.getenv("TABLE_EXTRACTOR", "pdfplumber").lower() == "camelot"
        )


def ensure_phase6_fields(settings: Settings) -> None:
    """Backfill Phase 6 visual retrieval attributes (Streamlit hot-reload)."""
    if not hasattr(settings, "enable_visual_retrieval"):
        settings.enable_visual_retrieval = (
            os.getenv("ENABLE_VISUAL_RETRIEVAL", "true").lower() == "true"
        )
    if not hasattr(settings, "visual_faiss_index_path"):
        settings.visual_faiss_index_path = os.getenv(
            "VISUAL_FAISS_INDEX_PATH", "data/parsed/faiss.index.visual"
        )
    if not hasattr(settings, "visual_embedding_model"):
        settings.visual_embedding_model = os.getenv(
            "VISUAL_EMBEDDING_MODEL", "clip-ViT-B-32"
        )
    if not hasattr(settings, "visual_top_k"):
        settings.visual_top_k = int(os.getenv("VISUAL_TOP_K", "20"))
    if not hasattr(settings, "visual_fusion_lambda"):
        settings.visual_fusion_lambda = float(
            os.getenv("VISUAL_FUSION_LAMBDA", "0.65")
        )
    if not hasattr(settings, "visual_device"):
        settings.visual_device = os.getenv("VISUAL_DEVICE", "")
    if not hasattr(settings, "visual_batch_size"):
        settings.visual_batch_size = int(os.getenv("VISUAL_BATCH_SIZE", "8"))
    if not hasattr(settings, "visual_for_image_intent_only"):
        settings.visual_for_image_intent_only = (
            os.getenv("VISUAL_FOR_IMAGE_INTENT_ONLY", "true").lower() == "true"
        )


def ensure_colpali_fields(settings: Settings) -> None:
    if not hasattr(settings, "enable_colpali_index"):
        settings.enable_colpali_index = (
            os.getenv("ENABLE_COLPALI_INDEX", "false").lower() == "true"
        )
    if not hasattr(settings, "enable_colpali_retrieval"):
        settings.enable_colpali_retrieval = (
            os.getenv("ENABLE_COLPALI_RETRIEVAL", "false").lower() == "true"
        )
    if not hasattr(settings, "colpali_model_id"):
        settings.colpali_model_id = os.getenv(
            "COLPALI_MODEL_ID", "vidore/colpali-v1.2"
        )
    if not hasattr(settings, "colpali_index_dir"):
        settings.colpali_index_dir = os.getenv(
            "COLPALI_INDEX_DIR", "data/parsed/colpali"
        )
    if not hasattr(settings, "colpali_page_dpi"):
        settings.colpali_page_dpi = float(os.getenv("COLPALI_PAGE_DPI", "150"))
    if not hasattr(settings, "colpali_max_index_pages"):
        settings.colpali_max_index_pages = int(
            os.getenv("COLPALI_MAX_INDEX_PAGES", "0")
        )
    if not hasattr(settings, "colpali_top_k"):
        settings.colpali_top_k = int(os.getenv("COLPALI_TOP_K", "3"))
    if not hasattr(settings, "colpali_device"):
        settings.colpali_device = os.getenv("COLPALI_DEVICE", "")
    if not hasattr(settings, "colpali_max_images_for_llm"):
        settings.colpali_max_images_for_llm = int(
            os.getenv("COLPALI_MAX_IMAGES_FOR_LLM", "4")
        )
