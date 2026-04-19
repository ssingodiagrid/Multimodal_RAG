"""Render retrieved RAG chunks (text, table, image) in Streamlit."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from configs.settings import Settings


def render_retrieved_chunk(
    chunk: dict,
    _settings: Settings,
    repo_root: Path | None = None,
) -> None:
    root = repo_root or Path(__file__).resolve().parent.parent
    chunk_id = chunk.get("chunk_id", "")
    page = chunk.get("page", "?")
    score = chunk.get("score", "")
    src = chunk.get("score_source", "")
    st.caption(f"`{chunk_id}` · p{page} · score={score} ({src})")

    mod = chunk.get("modality") or "text"
    if mod == "colpali_page":
        rk = chunk.get("colpali_rank")
        st.caption(
            f"ColPali MaxSim · rank {rk}" if rk is not None else "ColPali MaxSim page"
        )
        ap = chunk.get("asset_path")
        if ap:
            p = Path(ap)
            if not p.is_absolute():
                p = root / p
            if p.is_file():
                st.image(str(p), caption=(chunk.get("text") or "")[:500])
            else:
                st.warning(f"Page image not found: {p}")
                st.markdown(chunk.get("text") or "")
        else:
            st.markdown(chunk.get("text") or "")
    elif mod == "image" or chunk.get("asset_path"):
        ap = chunk.get("asset_path")
        if ap:
            p = Path(ap)
            if not p.is_absolute():
                p = root / p
            if p.is_file():
                st.image(str(p), caption=(chunk.get("text") or "")[:500])
            else:
                st.warning(f"Image file not found: {p}")
                st.markdown(chunk.get("text") or "")
        else:
            st.markdown(chunk.get("text") or "")
    elif mod == "table" and chunk.get("table_json"):
        tj = chunk["table_json"]
        headers = list(tj.get("headers") or [])
        rows = tj.get("rows") or []
        if rows:
            try:
                import pandas as pd

                ncols = max(len(r) for r in rows) if rows else 0
                hdrs = headers[:ncols] if headers else None
                if hdrs is None or len(hdrs) < ncols:
                    hdrs = [f"c{i}" for i in range(ncols)]
                norm_rows = [list(r) + [""] * (ncols - len(r)) for r in rows]
                st.dataframe(pd.DataFrame(norm_rows, columns=hdrs[:ncols]), use_container_width=True)
            except Exception:
                st.markdown("```\n" + (chunk.get("text") or "") + "\n```")
        else:
            st.markdown(chunk.get("text") or "")
    else:
        st.markdown(chunk.get("text") or "")

    with st.expander("Raw metadata", expanded=False):
        st.json(chunk)
