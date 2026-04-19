from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from retrieval.vector_store import FaissVectorStore


@dataclass
class CacheHit:
    answer: str
    context: list[dict]


class SemanticCache:
    """FAISS over normalized query embeddings; metadata holds full cache payload."""

    def __init__(
        self,
        dim: int,
        base_path: str,
        threshold: float,
        max_entries: int,
    ):
        self.dim = dim
        self.base_path = Path(base_path)
        self.threshold = threshold
        self.max_entries = max_entries
        self._vectors: list[list[float]] = []
        self._payloads: list[dict] = []
        self._store: FaissVectorStore | None = None
        self._load_from_disk()

    def _faiss_path(self) -> Path:
        return Path(str(self.base_path) + ".faiss")

    def _load_from_disk(self) -> None:
        if not self._faiss_path().is_file():
            self._rebuild_store()
            return
        try:
            tmp = FaissVectorStore.load(str(self.base_path))
            if tmp.dim != self.dim:
                self._vectors = []
                self._payloads = []
                self._rebuild_store()
                return
            self._vectors = []
            self._payloads = list(tmp._meta)
            for i in range(tmp.index.ntotal):
                self._vectors.append(tmp.index.reconstruct(i).tolist())
        except (OSError, ValueError, json.JSONDecodeError, RuntimeError):
            self._vectors = []
            self._payloads = []
        self._rebuild_store()

    def _rebuild_store(self) -> None:
        self._store = FaissVectorStore(self.dim)
        if self._vectors:
            self._store.add(self._vectors, self._payloads)

    def _persist(self) -> None:
        self.base_path.parent.mkdir(parents=True, exist_ok=True)
        self._rebuild_store()
        if self._store is not None:
            self._store.save(str(self.base_path))

    def lookup(self, query_vector: list[float], fingerprint: str) -> CacheHit | None:
        if self._store is None or self._store.index.ntotal == 0:
            return None
        results = self._store.search(query_vector, top_k=1)
        if not results:
            return None
        r = results[0]
        if float(r["score"]) < self.threshold:
            return None
        meta = r["metadata"]
        if meta.get("fingerprint") != fingerprint:
            return None
        return CacheHit(answer=meta["answer"], context=list(meta["context"]))

    def store(
        self,
        query_vector: list[float],
        fingerprint: str,
        query_text: str,
        answer: str,
        context: list[dict],
    ) -> None:
        payload = {
            "fingerprint": fingerprint,
            "query_text": query_text,
            "answer": answer,
            "context": context,
        }
        self._vectors.append(list(query_vector))
        self._payloads.append(payload)
        while len(self._vectors) > self.max_entries:
            self._vectors.pop(0)
            self._payloads.pop(0)
        self._persist()
