"""Cross-encoder reranking.

First-stage retrieval is fast and approximate: BM25 and bi-encoder embeddings
never let the query and the passage *see each other*.  A cross-encoder does —
it runs the query and one passage through a transformer together and emits a
single relevance score.  That is far more accurate and far too slow to run over
a whole corpus, which is exactly why it belongs in the second stage over a
shortlist.

Implemented with fastembed's ONNX cross-encoder so no PyTorch is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class RerankHit:
    index: int      # index into the candidate list handed in
    score: float


class Reranker:
    """Lazy ONNX cross-encoder.  ``enabled is False`` degrades to no-op."""

    def __init__(self, model_name: str | None = None, threads: int | None = None):
        self.name = model_name
        self.enabled = False
        self.error: str | None = None
        self._enc = None
        if model_name is None:
            return
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            kwargs: dict = {"model_name": model_name}
            if threads:
                kwargs["threads"] = threads
            self._enc = TextCrossEncoder(**kwargs)
            self.enabled = True
        except Exception as exc:  # noqa: BLE001 - reranking is optional by design
            self._enc = None
            self.enabled = False
            self.error = str(exc)

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        """Relevance score per document, in the order given.

        Failures degrade to neutral scores rather than raising: a reranker
        outage must not take down retrieval.  Neutral scores leave the fused
        order untouched, and the invariant suite's "reranker puts the most
        relevant passage first" check is what catches a silently broken model.
        """
        if not documents:
            return []
        if not self.enabled:
            return [0.0] * len(documents)
        docs = list(documents)
        try:
            raw = list(self._enc.rerank(query, docs))
        except TypeError:
            try:
                raw = list(self._enc.rerank(query, docs, batch_size=16))
            except Exception as exc:  # noqa: BLE001
                self.error = str(exc)
                return [0.0] * len(documents)
        except Exception as exc:  # noqa: BLE001
            self.error = str(exc)
            return [0.0] * len(documents)
        return [float(x) for x in raw]

    def rerank(
        self, query: str, documents: Sequence[str], order: Sequence[int] | None = None
    ) -> list[RerankHit]:
        """Score and sort.  ``order`` maps positions to original candidate ids."""
        scores = self.score(query, documents)
        idx = list(range(len(scores))) if order is None else list(order)
        paired = list(zip(idx, scores))
        paired.sort(key=lambda p: (-p[1], p[0]))
        return [RerankHit(i, s) for i, s in paired]
