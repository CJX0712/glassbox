"""Persistence: chunk metadata, embeddings and the FAISS vector index.

Two representations of the same vectors are kept on purpose:

* a **FAISS** ``IndexFlatIP`` — the production path, exact inner product over
  unit vectors, which is exact cosine similarity;
* a plain ``.npy`` matrix — the *independent* path used by the self-test to
  verify FAISS returns what brute-force numpy returns.  When a retrieval bug
  exists, having two implementations is what tells you *which* one is wrong.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np

from .text import Chunk


# --------------------------------------------------------------------------
# Chunk metadata
# --------------------------------------------------------------------------
def chunks_to_json(chunks: Sequence[Chunk]) -> str:
    return json.dumps([asdict(c) for c in chunks], ensure_ascii=False, indent=1)


def chunks_from_json(blob: str) -> list[Chunk]:
    return [Chunk(**d) for d in json.loads(blob)]


# --------------------------------------------------------------------------
# Vector index
# --------------------------------------------------------------------------
class VectorStore:
    """Exact inner-product index with a numpy mirror."""

    def __init__(self, dim: int, vectors: np.ndarray | None = None):
        self.dim = int(dim)
        self.matrix = (
            np.zeros((0, self.dim), dtype=np.float32)
            if vectors is None
            else np.asarray(vectors, dtype=np.float32)
        )
        self._faiss = None
        self._index = None
        self._rebuild()

    # -- internals --------------------------------------------------------
    def _rebuild(self) -> None:
        try:
            import faiss  # type: ignore

            self._faiss = faiss
            index = faiss.IndexFlatIP(self.dim)
            if self.matrix.size:
                index.add(self.matrix)
            self._index = index
        except Exception:  # noqa: BLE001 - numpy mirror is always available
            self._faiss = None
            self._index = None

    # -- api --------------------------------------------------------------
    def __len__(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def backend(self) -> str:
        return "faiss:IndexFlatIP" if self._index is not None else "numpy:matmul"

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.size == 0:
            return
        if vectors.shape[1] != self.dim:
            raise ValueError(f"dim mismatch: got {vectors.shape[1]}, expected {self.dim}")
        self.matrix = np.vstack([self.matrix, vectors]) if self.matrix.size else vectors
        self._rebuild()

    def search(self, query_vec: np.ndarray, k: int) -> tuple[list[float], list[int]]:
        """Top-``k`` by inner product.  Returns ``(scores, indices)``."""
        n = len(self)
        if n == 0 or k <= 0:
            return [], []
        k = min(k, n)
        q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        if self._index is not None:
            scores, ids = self._index.search(q, k)
            return [float(x) for x in scores[0]], [int(x) for x in ids[0]]
        sims = (self.matrix @ q[0])
        order = np.argsort(-sims)[:k]
        return [float(sims[i]) for i in order], [int(i) for i in order]

    def brute_force(self, query_vec: np.ndarray, k: int) -> tuple[list[float], list[int]]:
        """Reference implementation, always numpy — used by the self-test."""
        n = len(self)
        if n == 0 or k <= 0:
            return [], []
        k = min(k, n)
        sims = self.matrix @ np.asarray(query_vec, dtype=np.float32)
        order = np.argsort(-sims, kind="stable")[:k]
        return [float(sims[i]) for i in order], [int(i) for i in order]

    # -- persistence ------------------------------------------------------
    def save(self, npy_path: Path, faiss_path: Path | None = None) -> None:
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy_path, self.matrix)
        if faiss_path is not None and self._faiss is not None and self._index is not None:
            self._faiss.write_index(self._index, str(faiss_path))

    @classmethod
    def load(cls, npy_path: Path, dim: int) -> "VectorStore":
        if npy_path.exists():
            mat = np.load(npy_path)
        else:
            mat = np.zeros((0, dim), dtype=np.float32)
        return cls(dim, mat)
