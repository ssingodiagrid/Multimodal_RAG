"""Build and load Phase 6 visual FAISS index (image chunks only)."""

from __future__ import annotations

import logging
from pathlib import Path

from retrieval.vector_store import FaissVectorStore

logger = logging.getLogger(__name__)


def visual_meta_from_image_chunk(c: dict) -> dict:
    """FAISS metadata for an image chunk (aligned with text index metadata)."""
    m = {
        "chunk_id": c["chunk_id"],
        "text": c["text"],
        "page": c["page"],
        "doc_id": c["doc_id"],
        "modality": c.get("modality", "image"),
    }
    for k in ("table_id", "image_id", "asset_path", "table_json", "pdf_caption"):
        if c.get(k) is not None:
            m[k] = c[k]
    return m


def build_visual_faiss_index(
    chunks: list[dict],
    settings,
    repo_root: Path,
) -> FaissVectorStore | None:
    """
    Embed image assets with CLIP-class model; save separate FAISS at
    settings.visual_faiss_index_path. No-op if no image rows or paths missing.
    """
    from retrieval.visual_embedder import VisualEmbedder

    image_chunks = [
        c
        for c in chunks
        if c.get("modality") == "image" and c.get("asset_path")
    ]
    paths: list[Path] = []
    metas: list[dict] = []
    for c in image_chunks:
        ap = c["asset_path"]
        p = Path(ap)
        if not p.is_absolute():
            p = repo_root / p
        if p.is_file():
            paths.append(p)
            metas.append(visual_meta_from_image_chunk(c))
        else:
            logger.warning("Visual index skip missing asset: %s", p)
    if not paths:
        logger.info("Visual FAISS index not built (no image assets on disk).")
        return None
    raw_dev = getattr(settings, "visual_device", "") or ""
    dev = raw_dev.strip() or None
    ve = VisualEmbedder(settings.visual_embedding_model, dev)
    vectors = ve.embed_image_paths(paths)
    dim = len(vectors[0])
    vs = FaissVectorStore(dim)
    vs.add(vectors, metas)
    base = settings.visual_faiss_index_path
    vs.save(base)
    logger.info("Saved visual FAISS index %s (%d vectors)", base, len(vectors))
    return vs


def try_load_visual_store(settings) -> FaissVectorStore | None:
    if not getattr(settings, "enable_visual_retrieval", False):
        return None
    base = getattr(settings, "visual_faiss_index_path", "")
    if not base:
        return None
    faiss_file = Path(str(base) + ".faiss")
    if not faiss_file.is_file():
        return None
    try:
        return FaissVectorStore.load(base)
    except Exception as exc:
        logger.warning("Could not load visual FAISS index: %s", exc)
        return None


def should_run_visual_merge(settings, modality_intent: str | None) -> bool:
    if not getattr(settings, "enable_visual_retrieval", False):
        return False
    if getattr(settings, "visual_for_image_intent_only", True):
        return modality_intent in ("image", "mixed")
    return True


def visual_merge_gate_intent(settings, modality_intent: str | None, query: str) -> str | None:
    """
    Intent for should_run_visual_merge / CLIP query embed.
    When the modality router is off, modality_intent is None; use the same keyword
    heuristic so queries like "show the image" still trigger Phase 6.
    """
    if not getattr(settings, "visual_for_image_intent_only", True):
        return None
    if modality_intent is not None:
        return modality_intent
    from retrieval.modality_router import route_query_heuristic

    return route_query_heuristic(query)
