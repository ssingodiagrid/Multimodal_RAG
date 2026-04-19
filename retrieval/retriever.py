def retrieve_context(search_results: list[dict], top_k: int = 5) -> list[dict]:
    context = []
    for item in search_results[:top_k]:
        meta = item["metadata"]
        row = {
            "chunk_id": meta["chunk_id"],
            "text": meta["text"],
            "page": meta["page"],
            "score": item["score"],
            "score_source": "dense",
        }
        for k in ("modality", "asset_path", "table_json", "table_id", "image_id", "doc_id"):
            if k in meta and meta[k] is not None:
                row[k] = meta[k]
        context.append(row)
    return context
