"""Vector stores.

`LocalVectorStore` is a persistent, dependency-light store (NumPy matrix +
JSON metadata) with exact cosine search. It is ideal for development and for
corpora up to tens of thousands of chunks. `ChromaVectorStore` uses ChromaDB's
HNSW index for larger corpora (install `chromadb`; works on Linux/Cloud Run).
Both implement the same interface so the backend is a config switch.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Protocol

import numpy as np

from ..ingest.chunker import Chunk


class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...
    def delete_source(self, source: str) -> int: ...
    def search(self, query: list[float], k: int) -> list[tuple[Chunk, float]]: ...
    def all_chunks(self) -> list[Chunk]: ...
    def count(self) -> int: ...
    def clear(self) -> None: ...


class LocalVectorStore:
    def __init__(self, directory: Path | None = None):
        self.dir = Path(directory) if directory else None
        self._lock = threading.RLock()
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None
        self.version = 0
        self._load()

    # ---- persistence ------------------------------------------------------
    def _paths(self) -> tuple[Path, Path] | None:
        if self.dir is None:
            return None
        return self.dir / "vectors.npy", self.dir / "chunks.json"

    def _load(self) -> None:
        paths = self._paths()
        if not paths or not paths[0].exists() or not paths[1].exists():
            return
        self._matrix = np.load(paths[0])
        self._chunks = [Chunk(**d) for d in json.loads(paths[1].read_text(encoding="utf-8"))]
        if len(self._chunks) != (0 if self._matrix is None else self._matrix.shape[0]):
            raise RuntimeError("Vector store is corrupted: chunk count does not match vector count")

    def _save(self) -> None:
        paths = self._paths()
        if not paths:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        vec_path, meta_path = paths
        # Write to temp files first so a crash never leaves a half-written index.
        tmp_vec, tmp_meta = vec_path.with_suffix(".tmp.npy"), meta_path.with_suffix(".tmp")
        np.save(tmp_vec, self._matrix if self._matrix is not None else np.zeros((0, 0), dtype=np.float32))
        tmp_meta.write_text(json.dumps([c.to_dict() for c in self._chunks]), encoding="utf-8")
        tmp_vec.replace(vec_path)
        tmp_meta.replace(meta_path)

    # ---- mutations --------------------------------------------------------
    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return
        new = np.asarray(embeddings, dtype=np.float32)
        with self._lock:
            if self._matrix is not None and self._matrix.size and self._matrix.shape[1] != new.shape[1]:
                raise ValueError(
                    f"Embedding dimension changed ({self._matrix.shape[1]} -> {new.shape[1]}); "
                    "clear the index and re-ingest"
                )
            incoming = {c.id for c in chunks}
            keep = [i for i, c in enumerate(self._chunks) if c.id not in incoming]
            old = self._matrix[keep] if self._matrix is not None and keep else np.zeros((0, new.shape[1]), np.float32)
            self._chunks = [self._chunks[i] for i in keep] + list(chunks)
            self._matrix = np.vstack([old, new])
            self.version += 1
            self._save()

    def delete_source(self, source: str) -> int:
        with self._lock:
            keep = [i for i, c in enumerate(self._chunks) if c.source != source]
            removed = len(self._chunks) - len(keep)
            if removed:
                self._chunks = [self._chunks[i] for i in keep]
                self._matrix = self._matrix[keep] if self._matrix is not None else None
                self.version += 1
                self._save()
            return removed

    def clear(self) -> None:
        with self._lock:
            self._chunks, self._matrix = [], None
            self.version += 1
            self._save()

    # ---- queries ----------------------------------------------------------
    def search(self, query: list[float], k: int) -> list[tuple[Chunk, float]]:
        with self._lock:
            if self._matrix is None or not len(self._chunks):
                return []
            q = np.asarray(query, dtype=np.float32)
            q = q / (np.linalg.norm(q) or 1.0)
            norms = np.linalg.norm(self._matrix, axis=1)
            norms[norms == 0] = 1.0
            scores = (self._matrix @ q) / norms
            k = min(k, len(scores))
            top = np.argpartition(-scores, k - 1)[:k]
            top = top[np.argsort(-scores[top])]
            return [(self._chunks[i], float(scores[i])) for i in top]

    def all_chunks(self) -> list[Chunk]:
        with self._lock:
            return list(self._chunks)

    def count(self) -> int:
        return len(self._chunks)


class ChromaVectorStore:
    """ChromaDB-backed store (HNSW, cosine). Requires `pip install chromadb`."""

    def __init__(self, directory: Path, collection: str = "knowledge"):
        import chromadb

        self._client = chromadb.PersistentClient(path=str(directory))
        self._name = collection
        self._col = self._client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})
        self.version = 0

    def upsert(self, chunks, embeddings):
        if not chunks:
            return
        self._col.upsert(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, "section": c.section, "page": c.page or 0} for c in chunks],
        )
        self.version += 1

    def delete_source(self, source):
        before = self._col.count()
        self._col.delete(where={"source": source})
        self.version += 1
        return before - self._col.count()

    def search(self, query, k):
        if self._col.count() == 0:
            return []
        res = self._col.query(query_embeddings=[query], n_results=min(k, self._col.count()))
        out = []
        rows = zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0], strict=True)
        for id_, doc, meta, dist in rows:
            chunk = Chunk(id=id_, source=meta["source"], text=doc, section=meta.get("section", ""),
                          page=meta.get("page") or None)
            out.append((chunk, 1.0 - float(dist)))
        return out

    def all_chunks(self):
        res = self._col.get(include=["documents", "metadatas"])
        return [Chunk(id=i, source=m["source"], text=d, section=m.get("section", ""), page=m.get("page") or None)
                for i, d, m in zip(res["ids"], res["documents"], res["metadatas"], strict=True)]

    def count(self):
        return self._col.count()

    def clear(self):
        self._client.delete_collection(self._name)
        self._col = self._client.get_or_create_collection(self._name, metadata={"hnsw:space": "cosine"})
        self.version += 1
