"""Streamlit UI: index PDFs and query Phase 1 text RAG."""

import shutil
import sys
from io import BytesIO
from pathlib import Path

# Streamlit runs this file with ui/ on sys.path; project root must be importable.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from configs.settings import (
    Settings,
    ensure_colpali_fields,
    ensure_phase4_fields,
    ensure_phase5_fields,
    ensure_phase6_fields,
)
from main import index_pdf, load_index, run_rag_query
from retrieval.bm25 import save_sparse_corpus
from retrieval.embedder import Embedder
from ui.context_evidence import render_retrieved_chunk

st.set_page_config(page_title="Phase 1 Text RAG", layout="wide")


@st.cache_resource
def _embedder(model: str) -> Embedder:
    return Embedder(model)


def _init_chat_state() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def _render_chat_history(settings: Settings) -> None:
    for item in st.session_state.chat_history:
        with st.chat_message("user"):
            if item.get("had_query_image"):
                st.caption("Reference image was included with this question.")
            st.markdown(item["query"])
        with st.chat_message("assistant"):
            st.markdown(item["answer"])
            with st.expander("Retrieved context"):
                for i, ch in enumerate(item["context"]):
                    st.markdown(f"**Evidence {i + 1}**")
                    render_retrieved_chunk(ch, settings)
                    st.divider()


def _reset_indexes(settings: Settings) -> None:
    base = Path(settings.faiss_index_path)
    dense_files = [Path(str(base) + ".faiss"), Path(str(base) + ".meta.json")]
    sparse_file = Path(settings.sparse_index_path)
    cache_base = Path(settings.semantic_cache_path)
    cache_files = [
        Path(str(cache_base) + ".faiss"),
        Path(str(cache_base) + ".meta.json"),
    ]
    for p in dense_files + [sparse_file] + cache_files:
        if p.exists():
            p.unlink()
    vbase = Path(settings.visual_faiss_index_path)
    for ext in (".faiss", ".meta.json"):
        vp = Path(str(vbase) + ext)
        if vp.exists():
            vp.unlink()
    assets = Path(settings.assets_dir)
    if assets.exists():
        shutil.rmtree(assets)
    assets.mkdir(parents=True, exist_ok=True)
    cp = Path(settings.colpali_index_dir)
    if not cp.is_absolute():
        cp = Path(__file__).resolve().parent.parent / cp
    if cp.exists():
        shutil.rmtree(cp)


def main() -> None:
    settings = Settings()
    ensure_phase4_fields(settings)
    ensure_phase5_fields(settings)
    ensure_phase6_fields(settings)
    ensure_colpali_fields(settings)
    st.title("Phase 1 Text RAG")
    _init_chat_state()

    index_exists = Path(settings.faiss_index_path + ".faiss").exists()

    with st.sidebar:
        st.subheader("Corpus")
        uploaded = st.file_uploader("Upload PDF", type=["pdf"])
        if uploaded is not None and st.button("Index uploaded PDF"):
            raw_dir = Path("data/raw_pdf")
            raw_dir.mkdir(parents=True, exist_ok=True)
            out_path = raw_dir / uploaded.name
            out_path.write_bytes(uploaded.getvalue())
            with st.spinner("Indexing (embeddings may download on first run)..."):
                # Enforce single-PDF mode: wipe old dense/sparse artifacts first.
                _reset_indexes(settings)
                index_pdf(str(out_path), settings)
                # Rebuild sparse corpus from this fresh dense index only.
                store = load_index(settings)
                save_sparse_corpus(store.all_metadata(), settings.sparse_index_path)
            st.session_state.chat_history = []
            n_img = sum(
                1 for m in store.all_metadata() if m.get("modality") == "image"
            )
            st.success(
                f"Indexed {uploaded.name} — {n_img} image chunk(s) "
                "(figures / page renders appear under Retrieved context)."
            )
            if settings.enable_image_captions and n_img == 0:
                st.warning(
                    "No image chunks were extracted. Many PDFs need "
                    "`ENABLE_IMAGE_PAGE_RENDERS=true` (see `.env.example`) and a re-index."
                )
            st.rerun()
        if index_exists:
            st.caption(f"Index: `{settings.faiss_index_path}.*`")
            try:
                active_store = load_index(settings)
                doc_ids = sorted({m.get("doc_id", "") for m in active_store.all_metadata() if m.get("doc_id")})
                if doc_ids:
                    st.caption(f"Active corpus: `{', '.join(doc_ids[:3])}`")
            except Exception:
                pass
        else:
            st.info("Upload a PDF and click **Index uploaded PDF**.")
        if st.button("Clear chat history"):
            st.session_state.chat_history = []
            st.rerun()
        st.divider()
        st.subheader("Image query (optional)")
        st.caption(
            "CLIP search + Gemini vision: attach an image, then ask in the chat box."
        )
        qimg = st.file_uploader(
            "Image for next question",
            type=["png", "jpg", "jpeg", "webp"],
            key="query_image_uploader",
            label_visibility="collapsed",
        )
        if qimg is not None:
            st.session_state["pending_query_image"] = (
                qimg.getvalue(),
                (qimg.type or "").strip() or "image/png",
            )
        pending = st.session_state.get("pending_query_image")
        if pending:
            b, _mt = pending
            st.image(BytesIO(b), width=200, caption="Attached for your next message")
            if st.button("Remove attached image"):
                st.session_state.pop("pending_query_image", None)
                st.rerun()

    st.subheader("Conversation")
    if not st.session_state.chat_history:
        st.caption("No questions yet. Use the chat box at the bottom.")
    else:
        _render_chat_history(settings)

    query = st.chat_input("Ask your question")
    if query:
        if not index_exists:
            st.warning("Index a PDF first (sidebar).")
            return
        pending = st.session_state.get("pending_query_image")
        q_bytes, q_mime = (None, None)
        if pending:
            q_bytes, q_mime = pending[0], pending[1]
        embedder = _embedder(settings.embedding_model)
        store = load_index(settings)
        with st.spinner("Retrieving and generating..."):
            try:
                answer, context = run_rag_query(
                    query,
                    store,
                    embedder,
                    settings,
                    query_image_bytes=q_bytes,
                    query_image_mime=q_mime,
                )
                if pending:
                    st.session_state.pop("pending_query_image", None)
                display_q = (query or "").strip() or "[Image-only question]"
                st.session_state.chat_history.append(
                    {
                        "query": display_q,
                        "answer": answer,
                        "context": context,
                        "had_query_image": bool(q_bytes),
                    }
                )
                st.rerun()
            except Exception as e:
                st.error(str(e))
                if (
                    "GCP_PROJECT_ID" in str(e)
                    or "GOOGLE_APPLICATION_CREDENTIALS" in str(e)
                    or "vertex" in str(e).lower()
                ):
                    st.caption(
                        "Check `.env` for `GCP_PROJECT_ID`, `GCP_LOCATION`, and credentials path."
                    )


if __name__ == "__main__":
    main()
