"""LLM rewrite of user text into a short retrieval-oriented query (optional path)."""

from __future__ import annotations

import logging

from configs.settings import Settings
from generation.llm_pipeline import GeminiClient

logger = logging.getLogger(__name__)


def refine_search_query(user_query: str, settings: Settings) -> str | None:
    """
    Ask Gemini for a single search-style rewrite. Returns None if rewrite is empty,
    identical to input, or the call fails (caller should fall back to raw-only retrieval).
    """
    q = (user_query or "").strip()
    if len(q) < 3:
        return None
    prompt = (
        "Rewrite the following user message into ONE short standalone search query for "
        "keyword- and embedding-based document retrieval. Preserve named entities, "
        "numbers, and product or model codes. Do not answer the question; output only "
        "the rewritten query text with no quotes, labels, markdown, or explanation.\n\n"
        f"User message:\n{q}"
    )
    try:
        llm = GeminiClient(
            model_name=settings.gemini_model,
            project=settings.gcp_project_id or None,
            location=settings.gcp_location,
        )
        out = llm.answer(prompt).strip()
        for prefix in ("Rewritten query:", "Search query:", "Query:", "Rewritten:"):
            low = out.lower()
            if low.startswith(prefix.lower()):
                out = out[len(prefix) :].strip()
        if out.startswith('"') and out.endswith('"') and len(out) > 1:
            out = out[1:-1].strip()
        out = out.split("\n")[0].strip()
        if len(out) > 500:
            out = out[:500].rstrip()
        if not out or out.lower() == q.lower():
            return None
        return out
    except Exception as exc:
        logger.warning("Query refinement skipped: %s", exc)
        return None
