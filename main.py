"""Phase 1 text RAG: ingest PDF, index with FAISS, query with dense retrieval + Gemini."""

from __future__ import annotations

import logging
import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from cache.index_fingerprint import compute_index_fingerprint
from cache.semantic_cache import SemanticCache
from configs.settings import (
    Settings,
    ensure_colpali_fields,
    ensure_phase4_fields,
    ensure_phase5_fields,
    ensure_phase6_fields,
)
from generation.image_caption import caption_image_with_gemini, image_to_chunk
from generation.llm_pipeline import GeminiClient
from generation.prompt_builder import build_grounded_prompt
from generation.query_refinement import refine_search_query
from ingestion.chunker import chunk_pages
from ingestion.images_extract import extract_images_from_pdf
from ingestion.page_render_extract import extract_page_renders_from_pdf
from ingestion.pdf_loader import extract_pages
from ingestion.tables_extract import extract_tables_from_pdf
from ingestion.tables_normalize import table_to_chunks
from retrieval.dual_query_merge import merge_dual_retrieval_contexts
from retrieval.embedder import Embedder
from retrieval.modality_router import route_query
from retrieval.multihop import (
    generate_sub_query,
    merge_contexts,
    should_multihop,
)
from retrieval.pipeline import run_phase3_retrieval
from retrieval.retriever import retrieve_context
from retrieval.vector_store import FaissVectorStore

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parent


def _trace_event(name: str, metadata: dict | None = None) -> None:
    if not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"):
        return
    try:
        from langfuse import Langfuse

        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        lf = Langfuse(host=host)
        lf.trace(name=name, metadata=metadata or {})
    except Exception:
        pass


def _meta_from_chunk(c: dict) -> dict:
    m = {
        "chunk_id": c["chunk_id"],
        "text": c["text"],
        "page": c["page"],
        "doc_id": c["doc_id"],
        "modality": c.get("modality", "text"),
    }
    for k in ("table_id", "image_id", "asset_path", "table_json", "pdf_caption"):
        if c.get(k) is not None:
            m[k] = c[k]
    return m


def _relative_asset_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _primary_doc_id_from_store(store: FaissVectorStore) -> str | None:
    ids = sorted({m.get("doc_id", "") for m in store.all_metadata() if m.get("doc_id")})
    return ids[0] if ids else None


def _merge_colpali_into_context(
    base: list[dict],
    colpali_hits: list[dict],
    settings: Settings,
) -> list[dict]:
    if not colpali_hits:
        return base
    cap = int(settings.top_k) + int(getattr(settings, "colpali_top_k", 3)) + 4
    seen: set[tuple[str, int]] = set()
    out: list[dict] = []
    for c in colpali_hits + base:
        key = (str(c.get("chunk_id", "")), int(c.get("page") or -1))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out[:cap]


def index_pdf(pdf_path: str, settings: Settings | None = None) -> FaissVectorStore:
    settings = settings or Settings()
    ensure_phase5_fields(settings)
    ensure_phase6_fields(settings)
    ensure_colpali_fields(settings)
    path = Path(pdf_path)
    doc_id = path.stem
    pages = extract_pages(str(path))
    chunks = chunk_pages(
        doc_id, pages, settings.chunk_size, settings.chunk_overlap
    )
    for c in chunks:
        c["modality"] = "text"
    if settings.enable_table_extraction:
        try:
            raw_tables = extract_tables_from_pdf(
                str(path),
                prefer_camelot=settings.table_extractor_prefer_camelot,
            )
            for t in raw_tables:
                chunks.extend(
                    table_to_chunks(
                        doc_id,
                        t["page"],
                        t["table_index"],
                        t["headers"],
                        t["rows"],
                        settings.table_max_rows_per_chunk,
                    )
                )
        except Exception as exc:
            logger.warning("Table extraction skipped: %s", exc)
    if settings.enable_image_captions:
        try:
            images = extract_images_from_pdf(
                str(path),
                settings.assets_dir,
                doc_id,
                settings.image_caption_max_side,
            )
            for im in images:
                cap = ""
                try:
                    cap = caption_image_with_gemini(
                        im["path"],
                        im["mime"],
                        settings.gemini_model,
                        settings.gcp_project_id or None,
                        settings.gcp_location,
                    )
                except Exception as exc:
                    logger.warning("Image caption skipped: %s", exc)
                chunks.append(
                    image_to_chunk(
                        doc_id,
                        im["page"],
                        im["image_index"],
                        cap,
                        _relative_asset_path(im["path"]),
                        source="embedded",
                    )
                )
            if getattr(settings, "enable_image_page_renders", False):
                dpi = float(getattr(settings, "image_page_render_dpi", 120.0))
                strat = getattr(
                    settings, "image_page_render_strategy", "figures"
                )
                renders = extract_page_renders_from_pdf(
                    str(path),
                    settings.assets_dir,
                    doc_id,
                    dpi,
                    settings.image_caption_max_side,
                    strategy=strat,
                )
                for im in renders:
                    cap = ""
                    label = (im.get("label") or "").strip()
                    pdf_cap = (im.get("pdf_caption") or "").strip()
                    try:
                        cap = caption_image_with_gemini(
                            im["path"],
                            im["mime"],
                            settings.gemini_model,
                            settings.gcp_project_id or None,
                            settings.gcp_location,
                        )
                    except Exception as exc:
                        logger.warning("Page render caption skipped: %s", exc)
                    # PDF caption first so embeddings/BM25 match user queries that quote the doc.
                    if pdf_cap:
                        body = f"{pdf_cap} {cap}".strip() if cap else pdf_cap
                    else:
                        body = f"{label} {cap}".strip() if label else cap.strip()
                    if not body:
                        body = (
                            f"[Page {im['page']} render]"
                            if strat == "full_pages"
                            else f"[Figure region page {im['page']}]"
                        )
                    ch = image_to_chunk(
                        doc_id,
                        im["page"],
                        im["image_index"],
                        body,
                        _relative_asset_path(im["path"]),
                        source="page_render",
                    )
                    if pdf_cap:
                        ch["pdf_caption"] = pdf_cap
                    chunks.append(ch)
        except Exception as exc:
            logger.warning("Image extraction skipped: %s", exc)
    if not chunks:
        raise ValueError(f"No text extracted from {pdf_path}")
    embedder = Embedder(settings.embedding_model)
    texts = [c["text"] for c in chunks]
    vectors = embedder.embed_texts(texts)
    dim = len(vectors[0])
    store = FaissVectorStore(dim)
    meta = [_meta_from_chunk(c) for c in chunks]
    store.add(vectors, meta)
    base = settings.faiss_index_path
    store.save(base)
    if getattr(settings, "enable_visual_retrieval", False):
        try:
            from retrieval.visual_index import build_visual_faiss_index

            vis_out = build_visual_faiss_index(chunks, settings, REPO_ROOT)
            if vis_out is None:
                logger.warning(
                    "Visual FAISS not written (no on-disk image assets resolved, or empty "
                    "image chunk list). After fixing .env/assets, run: "
                    "python scripts/rebuild_visual_faiss.py"
                )
        except Exception as exc:
            logger.warning("Visual FAISS index skipped: %s", exc, exc_info=True)
    if getattr(settings, "enable_colpali_index", False):
        try:
            from retrieval.colpali_retrieval import build_colpali_page_index

            raw_cd = getattr(settings, "colpali_device", "") or ""
            cdev = raw_cd.strip() or None
            build_colpali_page_index(
                str(path),
                doc_id,
                repo_root=REPO_ROOT,
                assets_dir=settings.assets_dir,
                index_dir=settings.colpali_index_dir,
                model_id=settings.colpali_model_id,
                page_dpi=float(getattr(settings, "colpali_page_dpi", 150.0)),
                max_pages=int(getattr(settings, "colpali_max_index_pages", 0)),
                device=cdev,
            )
        except Exception as exc:
            logger.warning("ColPali page index skipped: %s", exc, exc_info=True)
    _trace_event("index_pdf", {"doc_id": doc_id, "chunks": len(chunks), "path": base})
    return store


def load_index(settings: Settings | None = None) -> FaissVectorStore:
    settings = settings or Settings()
    return FaissVectorStore.load(settings.faiss_index_path)


def _retrieve_context(
    query: str,
    query_vector: list[float],
    store: FaissVectorStore,
    settings: Settings,
    modality_intent: str | None = None,
    visual_store: FaissVectorStore | None = None,
    visual_query_vector: list[float] | None = None,
) -> list[dict]:
    try:
        return run_phase3_retrieval(
            query,
            query_vector,
            store,
            settings,
            modality_intent=modality_intent,
            visual_store=visual_store,
            visual_query_vector=visual_query_vector,
        )
    except Exception:
        raw = store.search(query_vector, top_k=settings.top_k)
        return retrieve_context(raw, top_k=settings.top_k)


def answer_query(
    query: str,
    context_chunks: list[dict],
    settings: Settings | None = None,
    *,
    repo_root: Path | None = None,
    query_image_bytes: bytes | None = None,
    query_image_mime: str | None = None,
    refined_query_text: str | None = None,
) -> str:
    settings = settings or Settings()
    root = repo_root or REPO_ROOT
    colpali_chunks = [c for c in context_chunks if c.get("modality") == "colpali_page"]
    other_chunks = [c for c in context_chunks if c.get("modality") != "colpali_page"]
    prompt = build_grounded_prompt(
        query,
        other_chunks + colpali_chunks,
        include_user_image=bool(query_image_bytes),
        include_colpali=bool(colpali_chunks) and not query_image_bytes,
        refined_query=refined_query_text,
    )
    llm = GeminiClient(
        model_name=settings.gemini_model,
        project=settings.gcp_project_id or None,
        location=settings.gcp_location,
    )
    _trace_event(
        "answer_query",
        {
            "query_len": len(query),
            "context_n": len(context_chunks),
            "multimodal": bool(query_image_bytes) or bool(colpali_chunks),
        },
    )
    if query_image_bytes:
        return llm.answer_with_image(
            prompt,
            query_image_bytes,
            query_image_mime or "image/png",
        )
    if colpali_chunks:
        imgs: list[tuple[bytes, str]] = []
        nmax = int(getattr(settings, "colpali_max_images_for_llm", 4))
        for c in colpali_chunks[:nmax]:
            ap = c.get("asset_path")
            if not ap:
                continue
            p = Path(ap)
            if not p.is_absolute():
                p = root / p
            if p.is_file():
                imgs.append((p.read_bytes(), "image/png"))
        if imgs:
            return llm.answer_with_images(prompt, imgs)
    return llm.answer(prompt)


def run_rag_query(
    query: str,
    store: FaissVectorStore,
    embedder: Embedder,
    settings: Settings | None = None,
    *,
    query_image_bytes: bytes | None = None,
    query_image_mime: str | None = None,
) -> tuple[str, list[dict]]:
    settings = settings or Settings()
    ensure_phase4_fields(settings)
    ensure_phase5_fields(settings)
    ensure_phase6_fields(settings)
    ensure_colpali_fields(settings)
    effective_query = (query or "").strip() or (
        "Describe the user's image and relate it to the document using the retrieved context."
    )
    modality_intent = (
        route_query(effective_query, settings)
        if settings.enable_modality_router
        else None
    )
    mi_retrieval: str | None = "image" if query_image_bytes else modality_intent
    refined_q: str | None = None
    if (
        getattr(settings, "enable_query_refinement", False)
        and not query_image_bytes
    ):
        refined_q = refine_search_query(effective_query, settings)
    qv_raw = embedder.embed_query(effective_query)
    qv_refined: list[float] | None = None
    if refined_q:
        try:
            qv_refined = embedder.embed_query(refined_q)
        except Exception as exc:
            logger.warning("Refined-query embedding failed: %s", exc)
            qv_refined = None
            refined_q = None
    dim = len(qv_raw)
    sparse_for_fp = settings.sparse_index_path if settings.enable_hybrid else None
    v_fp = (
        settings.visual_faiss_index_path
        if settings.enable_visual_retrieval
        else None
    )
    v_model = (
        settings.visual_embedding_model if settings.enable_visual_retrieval else None
    )
    fingerprint = compute_index_fingerprint(
        settings.faiss_index_path,
        sparse_for_fp,
        settings.embedding_model,
        visual_faiss_path=v_fp,
        visual_embedding_model=v_model,
    )

    visual_store = None
    visual_qv: list[float] | None = None
    if settings.enable_visual_retrieval:
        from retrieval.visual_embedder import VisualEmbedder
        from retrieval.visual_index import (
            should_run_visual_merge,
            try_load_visual_store,
            visual_merge_gate_intent,
        )

        visual_store = try_load_visual_store(settings)
        _visual_gate = visual_merge_gate_intent(
            settings, modality_intent, effective_query
        )
        if query_image_bytes:
            _visual_gate = "image"
        want_visual = visual_store and (
            query_image_bytes is not None
            or should_run_visual_merge(settings, _visual_gate)
        )
        if want_visual:
            try:
                raw_dev = settings.visual_device or ""
                vdev = raw_dev.strip() or None
                ve = VisualEmbedder(settings.visual_embedding_model, vdev)
                if query_image_bytes is not None:
                    visual_qv = ve.embed_image_bytes(query_image_bytes)
                else:
                    visual_qv = ve.embed_query(effective_query)
            except Exception as exc:
                logger.warning("Visual query embedding failed: %s", exc)
                visual_qv = None

    semantic_cache: SemanticCache | None = None
    if (
        settings.enable_semantic_cache
        and not query_image_bytes
        and not getattr(settings, "enable_colpali_retrieval", False)
        and not getattr(settings, "enable_query_refinement", False)
    ):
        try:
            semantic_cache = SemanticCache(
                dim=dim,
                base_path=settings.semantic_cache_path,
                threshold=settings.semantic_cache_threshold,
                max_entries=settings.semantic_cache_max_entries,
            )
            hit = semantic_cache.lookup(qv_raw, fingerprint)
            if hit is not None:
                _trace_event("run_rag_query", {"cache_hit": True})
                return hit.answer, hit.context
        except Exception:
            semantic_cache = None

    sub_q: str | None = None
    multihop_used = False
    use_multihop = (
        settings.enable_multi_hop
        and not query_image_bytes
        and should_multihop(effective_query, settings)
    )
    dual_cap = min(
        int(getattr(settings, "hybrid_top_n", 20)),
        max(int(settings.top_k) * 2, int(settings.multi_hop_merged_top_k)),
    )

    def _retrieve_effective(text: str, vec: list[float]) -> list[dict]:
        return _retrieve_context(
            text,
            vec,
            store,
            settings,
            modality_intent=mi_retrieval,
            visual_store=visual_store,
            visual_query_vector=visual_qv,
        )

    rq = refined_q
    rqv = qv_refined
    dual_ok = bool(rq and rqv)

    if use_multihop:
        ctx1_raw = _retrieve_effective(effective_query, qv_raw)
        if dual_ok:
            ctx1 = merge_dual_retrieval_contexts(
                ctx1_raw,
                _retrieve_effective(rq, rqv),  # type: ignore[arg-type]
                dual_cap,
            )
        else:
            ctx1 = ctx1_raw
        sub_q = generate_sub_query(
            effective_query, ctx1[: int(settings.top_k)], settings
        )
        if sub_q:
            qv2 = embedder.embed_query(sub_q)
            ctx2 = _retrieve_context(
                sub_q,
                qv2,
                store,
                settings,
                modality_intent=mi_retrieval,
                visual_store=visual_store,
                visual_query_vector=visual_qv,
            )
            context = merge_contexts(
                ctx1, ctx2, settings.multi_hop_merged_top_k
            )
            multihop_used = True
        else:
            context = ctx1
    else:
        ctx_raw = _retrieve_effective(effective_query, qv_raw)
        if dual_ok:
            context = merge_dual_retrieval_contexts(
                ctx_raw,
                _retrieve_effective(rq, rqv),  # type: ignore[arg-type]
                int(settings.top_k),
            )
        else:
            context = ctx_raw

    if getattr(settings, "enable_colpali_retrieval", False) and not query_image_bytes:
        doc_id = _primary_doc_id_from_store(store)
        if doc_id:
            try:
                from retrieval.colpali_retrieval import search_colpali_pages

                raw_cd = getattr(settings, "colpali_device", "") or ""
                cdev = raw_cd.strip() or None
                hits = search_colpali_pages(
                    effective_query,
                    doc_id,
                    repo_root=REPO_ROOT,
                    index_dir=settings.colpali_index_dir,
                    model_id=settings.colpali_model_id,
                    top_k=int(getattr(settings, "colpali_top_k", 3)),
                    device=cdev,
                )
                context = _merge_colpali_into_context(context, hits, settings)
            except Exception as exc:
                logger.warning("ColPali retrieval skipped: %s", exc)

    answer = answer_query(
        effective_query,
        context,
        settings,
        repo_root=REPO_ROOT,
        query_image_bytes=query_image_bytes,
        query_image_mime=query_image_mime,
        refined_query_text=rq if dual_ok else None,
    )
    trace_meta: dict = {"cache_hit": False, "multi_hop": multihop_used}
    if sub_q:
        trace_meta["sub_query"] = sub_q
    if dual_ok and rq:
        trace_meta["refined_query"] = rq
    _trace_event("run_rag_query", trace_meta)

    if (
        settings.enable_semantic_cache
        and semantic_cache is not None
        and not query_image_bytes
        and not getattr(settings, "enable_colpali_retrieval", False)
    ):
        try:
            semantic_cache.store(
                qv_raw, fingerprint, effective_query, answer, context
            )
        except Exception:
            pass

    return answer, context
