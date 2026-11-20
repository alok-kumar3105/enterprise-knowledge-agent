"""Hybrid retrieval: dense vectors + BM25 keywords, fused with Reciprocal Rank Fusion.

Dense search catches paraphrases ("time off" ~ "leave"); BM25 catches exact
terms such as policy codes, product names and numbers that embeddings blur.
RRF combines the two rankings without needing to calibrate their scores.
"""
from __future__ import annotations

import math
import threading
from collections import Counter
from dataclasses import dataclass

from ..ingest.chunker import Chunk
from ..llm import TracedLLM, tokenize
from .vector_store import VectorStore


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(d) for d in docs]
        self.lengths = [len(d) for d in docs]
        self.avgdl = (sum(self.lengths) / len(docs)) if docs else 0.0
        df: Counter = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for doc, dl in zip(self.docs, self.lengths, strict=True):
            s = 0.0
            for t in query:
                tf = doc.get(t)
                if tf:
                    denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                    s += self.idf[t] * tf * (self.k1 + 1) / denom
            out.append(s)
        return out


@dataclass
class Retrieved:
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None


def rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, id_ in enumerate(ranking):
            fused[id_] = fused.get(id_, 0.0) + 1.0 / (k + rank + 1)
    return fused


class HybridRetriever:
    def __init__(self, store: VectorStore, llm: TracedLLM, candidates: int = 20):
        self.store = store
        self.llm = llm
        self.candidates = candidates
        self._bm25: BM25 | None = None
        self._bm25_chunks: list[Chunk] = []
        self._bm25_version = -1
        self._lock = threading.Lock()

    def _ensure_bm25(self) -> None:
        version = getattr(self.store, "version", 0)
        with self._lock:
            if self._bm25 is None or self._bm25_version != version:
                self._bm25_chunks = self.store.all_chunks()
                self._bm25 = BM25([tokenize(f"{c.section} {c.text}") for c in self._bm25_chunks])
                self._bm25_version = version

    def search(self, query: str, k: int = 5) -> list[Retrieved]:
        if self.store.count() == 0:
            return []
        qvec = self.llm.embed([query], kind="query")[0]
        dense = self.store.search(qvec, self.candidates)

        self._ensure_bm25()
        sparse_scores = self._bm25.scores(tokenize(query))
        sparse_idx = sorted(range(len(sparse_scores)), key=lambda i: -sparse_scores[i])[: self.candidates]
        sparse = [(self._bm25_chunks[i], sparse_scores[i]) for i in sparse_idx if sparse_scores[i] > 0]

        dense_ids = [c.id for c, _ in dense]
        sparse_ids = [c.id for c, _ in sparse]
        fused = rrf([dense_ids, sparse_ids])
        by_id = {c.id: c for c, _ in dense} | {c.id: c for c, _ in sparse}
        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [
            Retrieved(
                chunk=by_id[id_],
                score=round(score, 5),
                dense_rank=dense_ids.index(id_) + 1 if id_ in dense_ids else None,
                sparse_rank=sparse_ids.index(id_) + 1 if id_ in sparse_ids else None,
            )
            for id_, score in ranked
        ]
