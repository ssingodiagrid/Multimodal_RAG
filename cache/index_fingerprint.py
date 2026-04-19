from __future__ import annotations

import hashlib
from pathlib import Path


def compute_index_fingerprint(
    faiss_path: str,
    sparse_path: str | None,
    embedding_model: str,
    visual_faiss_path: str | None = None,
    visual_embedding_model: str | None = None,
) -> str:
    """Stable fingerprint for corpus + embedder; changes when index files or model change."""
    parts: list[str] = [embedding_model]

    fp = Path(faiss_path)
    faiss_file = Path(str(fp) + ".faiss")
    if faiss_file.is_file():
        st = faiss_file.stat()
        parts.extend([str(faiss_file.resolve()), str(st.st_size), str(int(st.st_mtime_ns))])
    else:
        parts.append(f"missing_faiss:{faiss_path}")

    if sparse_path:
        sp = Path(sparse_path)
        if sp.is_file():
            st = sp.stat()
            parts.extend([str(sp.resolve()), str(st.st_size), str(int(st.st_mtime_ns))])
        else:
            parts.append(f"missing_sparse:{sparse_path}")

    if visual_faiss_path and visual_embedding_model:
        parts.append(str(visual_embedding_model))
        vfp = Path(visual_faiss_path)
        vf = Path(str(vfp) + ".faiss")
        if vf.is_file():
            st = vf.stat()
            parts.extend(
                [str(vf.resolve()), str(st.st_size), str(int(st.st_mtime_ns))]
            )
        else:
            parts.append(f"missing_visual_faiss:{visual_faiss_path}")

    raw = "|".join(parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
