from __future__ import annotations

_OPTIONAL_META = (
    "modality",
    "asset_path",
    "table_json",
    "table_id",
    "image_id",
)


def _meta_src(item: dict) -> dict:
    m = item.get("metadata")
    if isinstance(m, dict):
        return m
    return item


def _copy_optional_meta(src_item: dict, dest: dict) -> None:
    m = _meta_src(src_item)
    for k in _OPTIONAL_META:
        if k in m and m[k] is not None and dest.get(k) is None:
            dest[k] = m[k]


def _min_max_norm(values: list[float]) -> list[float]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if vmax - vmin < 1e-9:
        return [1.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


def fuse_dense_sparse(
    dense: list[dict], sparse: list[dict], alpha: float = 0.7, top_n: int = 20
) -> list[dict]:
    dense_scores = _min_max_norm([d["score"] for d in dense])
    sparse_scores = _min_max_norm([s["score"] for s in sparse])

    merged: dict[str, dict] = {}
    for item, norm_s in zip(dense, dense_scores):
        cid = item["metadata"]["chunk_id"] if "metadata" in item else item["chunk_id"]
        base = merged.setdefault(
            cid,
            {
                "chunk_id": cid,
                "text": item["metadata"]["text"] if "metadata" in item else item.get("text", ""),
                "page": item["metadata"]["page"] if "metadata" in item else item.get("page", -1),
                "doc_id": item["metadata"].get("doc_id", "")
                if "metadata" in item
                else item.get("doc_id", ""),
                "dense_score": 0.0,
                "sparse_score": 0.0,
            },
        )
        base["dense_score"] = float(norm_s)
        _copy_optional_meta(item, base)

    for item, norm_s in zip(sparse, sparse_scores):
        cid = item["chunk_id"]
        base = merged.setdefault(
            cid,
            {
                "chunk_id": cid,
                "text": item.get("text", ""),
                "page": item.get("page", -1),
                "doc_id": item.get("doc_id", ""),
                "dense_score": 0.0,
                "sparse_score": 0.0,
            },
        )
        base["sparse_score"] = float(norm_s)
        _copy_optional_meta(item, base)

    out = []
    for _, item in merged.items():
        item["hybrid_score"] = alpha * item["dense_score"] + (1.0 - alpha) * item["sparse_score"]
        out.append(item)
    out.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return out[:top_n]

