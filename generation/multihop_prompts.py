def build_sub_query_prompt(user_query: str, context_chunks: list[dict]) -> str:
    lines = []
    for i, c in enumerate(context_chunks[:8], start=1):
        t = (c.get("text") or "")[:600]
        lines.append(f"[{i}] (p{c.get('page', '?')}) {t}")
    ctx_block = "\n".join(lines) if lines else "(no evidence)"
    return f"""You refine retrieval for a RAG system. Given the user question and initial evidence snippets, output ONE short search query (5-15 words) to retrieve missing facts. Output ONLY valid JSON, no markdown, no explanation.

User question: {user_query}

Initial evidence:
{ctx_block}

JSON schema: {{"sub_query": "<string>"}}
"""
