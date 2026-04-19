from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import vertexai
from vertexai.generative_models import GenerativeModel, Part

from generation.llm_pipeline import _ensure_google_application_credentials


def caption_image_with_gemini(
    image_path: Path,
    mime: str,
    model_name: str,
    project: str | None,
    location: str,
) -> str:
    """Vertex Gemini multimodal: short caption for RAG indexing."""
    _ensure_google_application_credentials()
    proj = project or os.getenv("GCP_PROJECT_ID")
    loc = location or os.getenv("GCP_LOCATION", "us-central1")
    if not proj:
        raise ValueError("GCP_PROJECT_ID is required for image captioning.")
    vertexai.init(project=proj, location=loc)
    model = GenerativeModel(model_name)
    data = Path(image_path).read_bytes()
    part = Part.from_data(data=data, mime_type=mime or "image/png")
    prompt = (
        "Describe this document image (chart, figure, diagram, or photo) in 2-4 short "
        "sentences for search indexing. Focus on labels, trends, and subject matter."
    )
    resp: Any = model.generate_content([prompt, part])
    if hasattr(resp, "text") and resp.text:
        return resp.text.strip()
    if getattr(resp, "candidates", None):
        parts: list[str] = []
        for c in resp.candidates:
            content = getattr(c, "content", None)
            if content and getattr(content, "parts", None):
                for p in content.parts:
                    t = getattr(p, "text", None)
                    if t:
                        parts.append(t)
        if parts:
            return "".join(parts).strip()
    return ""


def image_to_chunk(
    doc_id: str,
    page: int,
    image_index: int,
    caption: str,
    asset_path_posix: str,
    *,
    source: str = "embedded",
) -> dict:
    """source: embedded (xref image) | page_render (pixmap crop or full page)."""
    tag = "img" if source == "embedded" else "render"
    return {
        "chunk_id": f"{doc_id}_p{page}_{tag}{image_index}",
        "doc_id": doc_id,
        "page": page,
        "text": caption.strip() or f"[Image on page {page}]",
        "modality": "image",
        "image_id": image_index,
        "asset_path": asset_path_posix,
    }
