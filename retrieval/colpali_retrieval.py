"""
ColPali-style late interaction (MaxSim) over full-page document images.

Requires: transformers with ColPali, torch, HuggingFace model download (e.g. vidore/colpali-v1.2).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_COLPALI_MODEL_KEYS = frozenset(
    {"input_ids", "attention_mask", "pixel_values", "token_type_ids"}
)


def _colpali_model_inputs(batch) -> dict:
    """Strip labels / extras ColPali VLM does not accept."""
    return {k: v for k, v in batch.items() if k in _COLPALI_MODEL_KEYS and v is not None}


def _index_root(repo_root: Path, index_dir: str) -> Path:
    p = Path(index_dir)
    return p if p.is_absolute() else (repo_root / p)


def colpali_manifest_path(repo_root: Path, doc_id: str, index_dir: str) -> Path:
    return _index_root(repo_root, index_dir) / doc_id / "manifest.json"


def build_colpali_page_index(
    pdf_path: str,
    doc_id: str,
    *,
    repo_root: Path,
    assets_dir: str,
    index_dir: str,
    model_id: str,
    page_dpi: float = 150.0,
    max_pages: int = 0,
    device: str | None = None,
) -> Path | None:
    """
    Rasterize pages, embed each with ColPali, save float32 tensors + manifest under index_dir/doc_id/.
    Page PNGs live under assets_dir/doc_id/ for reuse by the UI / Gemini.
    """
    from ingestion.colpali_raster import rasterize_pdf_pages_to_dir

    try:
        import torch
        from transformers import ColPaliForRetrieval, ColPaliProcessor
    except ImportError as exc:
        logger.warning("ColPali index skipped (missing deps): %s", exc)
        return None

    assets_base = repo_root / Path(assets_dir) / doc_id
    assets_base.mkdir(parents=True, exist_ok=True)
    page_paths = rasterize_pdf_pages_to_dir(
        pdf_path, assets_base, dpi=page_dpi, max_pages=max_pages
    )
    if not page_paths:
        logger.warning("ColPali: no pages rasterized for doc_id=%s", doc_id)
        return None

    out_root = _index_root(repo_root, index_dir) / doc_id
    out_root.mkdir(parents=True, exist_ok=True)

    from retrieval.torch_device import resolve_torch_device

    dev = resolve_torch_device(device)
    proc = ColPaliProcessor.from_pretrained(model_id)
    model = ColPaliForRetrieval.from_pretrained(model_id).to(dev)
    model.eval()

    pages_meta: list[dict] = []
    emb_dim = 0
    with torch.no_grad():
        for page_num, abs_png in page_paths:
            from PIL import Image

            im = Image.open(abs_png).convert("RGB")
            batch = proc.process_images(images=[im], return_tensors="pt").to(dev)
            out = model(**_colpali_model_inputs(batch))
            emb = out.embeddings[0].float().cpu()
            emb_dim = int(emb.shape[-1])
            stem = f"page_{page_num:04d}.pt"
            torch.save({"embeddings": emb, "page": page_num}, out_root / stem)
            try:
                rel_img = abs_png.relative_to(repo_root).as_posix()
            except ValueError:
                rel_img = abs_png.as_posix()
            pages_meta.append(
                {
                    "page": page_num,
                    "embeddings": stem,
                    "image": rel_img,
                }
            )

    manifest = {
        "doc_id": doc_id,
        "model": model_id,
        "embedding_dim": emb_dim,
        "pages": pages_meta,
    }
    man_path = out_root / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("ColPali index wrote %s (%d pages)", man_path, len(pages_meta))
    return man_path


def _load_manifest(repo_root: Path, index_dir: str, doc_id: str) -> dict | None:
    p = colpali_manifest_path(repo_root, doc_id, index_dir)
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _torch_load_compat(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def search_colpali_pages(
    query: str,
    doc_id: str,
    *,
    repo_root: Path,
    index_dir: str,
    model_id: str,
    top_k: int = 3,
    device: str | None = None,
) -> list[dict]:
    """
    MaxSim rank pages for query. Returns chunk-like dicts for merging into RAG context.
    """
    try:
        import torch
        from transformers import ColPaliForRetrieval, ColPaliProcessor
    except ImportError:
        return []

    man = _load_manifest(repo_root, index_dir, doc_id)
    if not man or not man.get("pages"):
        return []

    mid = man.get("model") or model_id
    from retrieval.torch_device import resolve_torch_device

    dev = resolve_torch_device(device)
    proc = ColPaliProcessor.from_pretrained(mid)
    model = ColPaliForRetrieval.from_pretrained(mid).to(dev)
    model.eval()

    root = _index_root(repo_root, index_dir) / doc_id
    passages: list[torch.Tensor] = []
    meta_rows: list[dict] = []
    for row in man["pages"]:
        pt_path = root / row["embeddings"]
        if not pt_path.is_file():
            continue
        blob = _torch_load_compat(pt_path)
        emb = blob["embeddings"].to(dev).to(model.dtype)
        passages.append(emb)
        meta_rows.append(row)

    if not passages:
        return []

    with torch.no_grad():
        qb = proc.process_queries([query], return_tensors="pt").to(dev)
        qout = model(**_colpali_model_inputs(qb))
        qemb = qout.embeddings[0].to(dtype=passages[0].dtype)

    scores = proc.score_retrieval([qemb], passages, batch_size=8, output_device="cpu")
    flat = scores[0].tolist()
    order = sorted(range(len(flat)), key=lambda i: flat[i], reverse=True)[: max(1, top_k)]

    out: list[dict] = []
    for rank, idx in enumerate(order):
        row = meta_rows[idx]
        score = float(flat[idx])
        page = int(row["page"])
        rel_img = row["image"]
        out.append(
            {
                "chunk_id": f"{doc_id}_colpali_p{page}",
                "doc_id": doc_id,
                "page": page,
                "text": (
                    f"[ColPali MaxSim page {page}] score={score:.4f}. "
                    f"Full-page visual match (see image evidence)."
                ),
                "modality": "colpali_page",
                "asset_path": rel_img,
                "score": score,
                "score_source": "colpali_maxsim",
                "colpali_rank": rank + 1,
            }
        )
    return out
