def _format_context_line(c: dict) -> str:
    page = c.get("page", "?")
    mod = c.get("modality") or "text"
    text = c.get("text") or ""
    if mod == "table":
        return f"[table p{page}] {text}"
    if mod == "image":
        ap = c.get("asset_path") or ""
        suffix = f" ({ap})" if ap else ""
        return f"[figure p{page}]{suffix} {text}"
    return f"[p{page}] {text}"


def build_grounded_prompt(
    query: str,
    context_chunks: list[dict],
    *,
    include_user_image: bool = False,
    include_colpali: bool = False,
    refined_query: str | None = None,
) -> str:
    context_text = "\n".join(_format_context_line(c) for c in context_chunks)
    has_figure = any((c.get("modality") or "") == "image" for c in context_chunks)
    figure_hint = ""
    if has_figure:
        figure_hint = (
            "Some lines are [figure pN] with text transcribed or captioned from document "
            "images (charts, screenshots, diagrams). Use that text as evidence about what "
            "the figure shows; cite (pN).\n"
        )
    user_image_hint = ""
    if include_user_image:
        user_image_hint = (
            "The user attached a reference image (you receive it as image input in addition "
            "to this text). Use it together with the retrieved context: compare or describe "
            "it when asked, and ground factual claims about the document in the context below "
            "with citations (pX). If a fact is not in the context, say I don't know for that part.\n"
        )
    colpali_hint = ""
    if include_colpali:
        colpali_hint = (
            "Following the text context, you will receive full-page images ranked by ColPali "
            "MaxSim (late interaction) as visual evidence. Cite page numbers (pN) when you use "
            "information you read from those page images.\n"
        )
    refinement_hint = ""
    if refined_query:
        refinement_hint = (
            "Retrieval used both the user's original wording and an LLM-rewritten search query; "
            "evidence below may reflect either. Answer the user's original question.\n"
            f"Rewritten search query used for retrieval: {refined_query}\n"
        )
    return (
        "You are a grounded assistant.\n"
        "Use only the provided context for document-specific facts.\n"
        "If the answer is not in context, respond with: I don't know.\n"
        f"{figure_hint}"
        f"{user_image_hint}"
        f"{colpali_hint}"
        f"{refinement_hint}\n"
        f"Question: {query}\n\n"
        f"Context:\n{context_text}\n\n"
        "Answer with citations like (pX) where the context supports them."
    )
