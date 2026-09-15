"""Dense retrieval — sentence embeddings through ONNX Runtime.

We deliberately use **fastembed** rather than sentence-transformers: it runs
the same checkpoints through ONNX Runtime instead of PyTorch.  On a CPU-only
laptop that is the difference between a ~60 MB install and a ~2.5 GB one, and
it keeps resident memory low enough to co-exist with a local LLM.

Vectors are L2-normalised on the way out, so *cosine similarity is exactly the
inner product* and the whole dense stage reduces to a single matrix multiply.
That identity is asserted in the self-test.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from .config import resolve_embed_model


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation; zero rows are left as zeros."""
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return mat / norms


class Embedder:
    """Thin, lazy wrapper around ``fastembed.TextEmbedding``."""

    def __init__(self, model_name: str | None = None, threads: int | None = None):
        from fastembed import TextEmbedding

        if model_name:
            self.name, self.declared_dim = model_name, 0
        else:
            self.name, self.declared_dim = resolve_embed_model()

        kwargs: dict = {"model_name": self.name}
        if threads:
            kwargs["threads"] = threads
        self.model = TextEmbedding(**kwargs)

        # Probe once to learn the true dimension (the registry can be stale).
        probe = np.asarray(next(iter(self.model.embed(["dimension probe"]))), dtype=np.float32)
        self.dim = int(probe.shape[0])

        # fastembed applies model-specific query/passage prefixes (e.g. the
        # "query: " / "passage: " convention for E5) only through these two
        # methods.  Fall back to plain embed() on older builds.
        self._has_split_api = hasattr(self.model, "query_embed") and hasattr(
            self.model, "passage_embed"
        )

    # -- encoding ---------------------------------------------------------
    def _collect(self, it: Iterable) -> np.ndarray:
        rows = [np.asarray(v, dtype=np.float32) for v in it]
        if not rows:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack(rows)

    def encode_documents(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        """Embed passages.  Returns ``(n, dim)`` float32, unit-norm rows."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self._has_split_api:
            it = self.model.passage_embed(list(texts), batch_size=batch_size)
        else:
            it = self.model.embed(list(texts), batch_size=batch_size)
        return l2_normalize(self._collect(it))

    def encode_query(self, query: str) -> np.ndarray:
        """Embed a single query.  Returns a ``(dim,)`` unit vector."""
        if self._has_split_api:
            it = self.model.query_embed([query])
        else:
            it = self.model.embed([query])
        return l2_normalize(self._collect(it))[0]

    def encode_queries(self, queries: Sequence[str], batch_size: int = 32) -> np.ndarray:
        if not queries:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self._has_split_api:
            it = self.model.query_embed(list(queries), batch_size=batch_size)
        else:
            it = self.model.embed(list(queries), batch_size=batch_size)
        return l2_normalize(self._collect(it))

    # -- scoring ----------------------------------------------------------
    def similarities(self, query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Cosine similarity of one query against a matrix of unit vectors."""
        if matrix.size == 0:
            return np.zeros((0,), dtype=np.float32)
        return matrix @ np.asarray(query_vec, dtype=np.float32)
