from __future__ import annotations

import json
import re

from configs.settings import Settings

ModalityIntent = str  # "text" | "table" | "image" | "mixed"


def route_query_heuristic(query: str) -> ModalityIntent:
    q = query.lower()
    table_kw = (
        "table",
        "tabular",
        "row",
        "column",
        "spreadsheet",
        "million",
        "billion",
        "revenue breakdown",
        "fy20",
        "q1",
        "quarter",
    )
    image_kw = (
        "figure",
        "fig.",
        "chart",
        "graph",
        "diagram",
        "image",
        "picture",
        "illustration",
        "plot",
        "shows a",
    )
    has_t = any(k in q for k in table_kw)
    has_i = any(k in q for k in image_kw)
    if has_t and has_i:
        return "mixed"
    if has_t:
        return "table"
    if has_i:
        return "image"
    return "text"


def route_query_llm(query: str, settings: Settings) -> ModalityIntent:
    from generation.llm_pipeline import GeminiClient

    prompt = (
        "Classify the user's query for document RAG retrieval.\n"
        'Return ONLY valid JSON: {"intent":"text"} or {"intent":"table"} or '
        '{"intent":"image"} or {"intent":"mixed"}.\n'
        "- text: general prose facts\n"
        "- table: numbers in tables, rows/columns\n"
        "- image: charts, figures, diagrams, visuals\n"
        "- mixed: both tabular and visual\n\n"
        f"Query: {query}"
    )
    llm = GeminiClient(
        model_name=settings.gemini_model,
        project=settings.gcp_project_id or None,
        location=settings.gcp_location,
    )
    raw = llm.answer(prompt).strip()
    m = re.search(r"\{[^}]+\}", raw)
    if m:
        raw = m.group(0)
    try:
        obj = json.loads(raw)
        intent = str(obj.get("intent", "text")).lower()
        if intent in ("text", "table", "image", "mixed"):
            return intent
    except json.JSONDecodeError:
        pass
    return "text"


def route_query(query: str, settings: Settings) -> ModalityIntent:
    """Caller should only use the result when `settings.enable_modality_router` is True."""
    if getattr(settings, "router_use_llm", False):
        try:
            return route_query_llm(query, settings)
        except Exception:
            return route_query_heuristic(query)
    return route_query_heuristic(query)
