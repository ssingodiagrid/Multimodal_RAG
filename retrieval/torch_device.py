"""Pick a PyTorch device string: MPS (Apple Silicon) before CUDA, then CPU."""

from __future__ import annotations


def resolve_torch_device(explicit: str | None = None) -> str:
    """
    Return one of: ``mps``, ``cuda``, ``cpu``.

    - If ``explicit`` is ``mps`` / ``cuda`` / ``cpu`` and available, use it.
    - Otherwise auto: **MPS first** (Mac GPU), then CUDA, then CPU.
    """
    import torch

    raw = (explicit or "").strip().lower()
    if raw == "cpu":
        return "cpu"
    if raw == "mps":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return resolve_torch_device(None)
    if raw == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        return resolve_torch_device(None)
    # Auto: Apple Silicon GPU first
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
