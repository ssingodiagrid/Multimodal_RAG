from __future__ import annotations

import json
from pathlib import Path

try:
    from rank_bm25 import BM25Okapi
except Exception:  # pragma: no cover
    BM25Okapi = None


def _tokenize(text: str) -> list[str]:
    return (text or "").lower().split()


class SimpleBM25Retriever:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.corpus_tokens = [_tokenize(c.get("text", "")) for c in chunks]
        self._bm25 = BM25Okapi(self.corpus_tokens) if BM25Okapi else None

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        q_tokens = _tokenize(query)
        if self._bm25 is not None:
            scores = self._bm25.get_scores(q_tokens)
        else:
            # Fallback lexical overlap scorer
            q_set = set(q_tokens)
            scores = [
                float(len(q_set.intersection(set(tokens))))
                for tokens in self.corpus_tokens
            ]
        ranked = sorted(
            zip(self.chunks, scores), key=lambda x: float(x[1]), reverse=True
        )[:top_k]
        out = []
        for chunk, score in ranked:
            out.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "page": chunk["page"],
                    "doc_id": chunk.get("doc_id", ""),
                    "score": float(score),
                }
            )
        return out


def save_sparse_corpus(chunks: list[dict], path: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    return str(p)


def load_sparse_corpus(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

