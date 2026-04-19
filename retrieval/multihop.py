from __future__ import annotations

import json
import re

from configs.settings import Settings
from generation.llm_pipeline import GeminiClient
from generation.multihop_prompts import build_sub_query_prompt


def parse_sub_query_json(text: str) -> str | None:
    raw = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if m:
        raw = m.group(1)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    sq = obj.get("sub_query")
    if isinstance(sq, str) and sq.strip():
        return sq.strip()
    return None


def should_multihop(query: str, settings: Settings) -> bool:
    if not settings.enable_multi_hop:
        return False
    mode = settings.multi_hop_mode.lower()
    if mode == "off":
        return False
    if mode == "always":
        return True
    q = query.lower()
    keys = (
        "compare ",
        "comparison",
        "versus",
        " vs.",
        " vs ",
        "difference between",
        "both ",
        "relationship between",
        "how do ",
        " and ",
        " as well as ",
    )
    return any(k in q for k in keys)


def merge_contexts(
    a: list[dict], b: list[dict], max_chunks: int
) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for chunk in a + b:
        cid = chunk.get("chunk_id")
        if cid is None or cid in seen:
            continue
        seen.add(str(cid))
        out.append(chunk)
        if len(out) >= max_chunks:
            break
    return out


def generate_sub_query(
    query: str, context_chunks: list[dict], settings: Settings
) -> str | None:
    prompt = build_sub_query_prompt(query, context_chunks)
    llm = GeminiClient(
        model_name=settings.gemini_model,
        project=settings.gcp_project_id or None,
        location=settings.gcp_location,
    )
    raw = llm.answer(prompt)
    return parse_sub_query_json(raw)
