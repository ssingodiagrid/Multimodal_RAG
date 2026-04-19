import json
import math
from pathlib import Path

import faiss
import numpy as np


class InMemoryVectorStore:
    def __init__(self):
        self._vectors: list[list[float]] = []
        self._meta: list[dict] = []

    def add(self, vectors: list[list[float]], metadata: list[dict]) -> None:
        self._vectors.extend(vectors)
        self._meta.extend(metadata)

    def search(self, query_vector: list[float], top_k: int = 5) -> list[dict]:
        scored = []
        for v, m in zip(self._vectors, self._meta):
            dot = sum(a * b for a, b in zip(query_vector, v))
            nq = math.sqrt(sum(a * a for a in query_vector))
            nv = math.sqrt(sum(a * a for a in v))
            score = dot / (nq * nv + 1e-9)
            scored.append({"score": score, "metadata": m})
        return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]


class FaissVectorStore:
    """Inner-product index for L2-normalized embeddings (cosine similarity)."""

    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self._meta: list[dict] = []

    def add(self, vectors: list[list[float]], metadata: list[dict]) -> None:
        if not vectors:
            return
        x = np.asarray(vectors, dtype=np.float32)
        self.index.add(x)
        self._meta.extend(metadata)

    def search(self, query_vector: list[float], top_k: int = 5) -> list[dict]:
        if self.index.ntotal == 0:
            return []
        q = np.asarray([query_vector], dtype=np.float32)
        scores, indices = self.index.search(q, min(top_k, self.index.ntotal))
        out: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            out.append({"score": float(score), "metadata": self._meta[idx]})
        return out

    def all_metadata(self) -> list[dict]:
        return list(self._meta)

    def save(self, base_path: str) -> None:
        p = Path(base_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(p) + ".faiss")
        with open(str(p) + ".meta.json", "w", encoding="utf-8") as f:
            json.dump(self._meta, f, ensure_ascii=False)

    @classmethod
    def load(cls, base_path: str) -> "FaissVectorStore":
        p = Path(base_path)
        index = faiss.read_index(str(p) + ".faiss")
        with open(str(p) + ".meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        store = cls(dim=index.d)
        store.index = index
        store._meta = meta
        return store
