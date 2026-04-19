"""Lazy-loaded CLIP-style image + text encoder (sentence-transformers). Phase 6."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _mitigate_native_thread_crash() -> None:
    """
    Reduce macOS/Linux segfaults during CLIP image encode (OpenMP + PyTorch + tokenizers).
    Call before importing torch / sentence_transformers.
    """
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    if sys.platform == "darwin":
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


_mitigate_native_thread_crash()


class VisualEmbedder:
    """CLIP-class dual encoder; loads model on first use."""

    def __init__(self, model_name: str, device: str | None = None):
        self.model_name = model_name
        self.device = device
        self._model = None

    def _model_device(self) -> str:
        try:
            from retrieval.torch_device import resolve_torch_device

            return resolve_torch_device(self.device)
        except ImportError:
            return "cpu"

    def _get_model(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            torch.set_num_threads(1)
            try:
                torch.set_num_interop_threads(1)
            except Exception:
                pass
            dev = self._model_device()
            logger.info("Loading visual embedding model %s on %s", self.model_name, dev)
            self._model = SentenceTransformer(self.model_name, device=dev)
        return self._model

    def embed_query(self, text: str) -> list[float]:
        m = self._get_model()
        v = m.encode(
            [text],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return [float(x) for x in v.tolist()]

    def embed_image_bytes(self, data: bytes) -> list[float]:
        """Single image from raw bytes (e.g. Streamlit upload)."""
        from io import BytesIO

        from PIL import Image

        import numpy as np

        m = self._get_model()
        im = Image.open(BytesIO(data)).convert("RGB")
        vecs = m.encode(
            [im],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        arr = np.asarray(vecs)
        if arr.ndim == 1:
            return [float(x) for x in arr.tolist()]
        return [float(x) for x in arr[0].tolist()]

    def embed_image_paths(self, paths: list[Path]) -> list[list[float]]:
        """Encode images one forward per image (avoids native crashes from batched CLIP vision on CPU/macOS)."""
        from PIL import Image

        import numpy as np

        m = self._get_model()
        images: list = []
        for p in paths:
            images.append(Image.open(p).convert("RGB"))
        if not images:
            return []
        # Single-image encodes avoid many native crashes on Darwin when batching CLIP vision.
        out: list[list[float]] = []
        for im in images:
            vecs = m.encode(
                [im],
                batch_size=1,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            arr = np.asarray(vecs)
            if arr.ndim == 1:
                out.append([float(x) for x in arr.tolist()])
            else:
                out.append([float(x) for x in arr[0].tolist()])
        return out
