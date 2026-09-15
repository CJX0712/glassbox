"""Sparse lexical retrieval — Okapi BM25.

BM25 is still the strongest cheap baseline for exact-match queries (product
codes, names, error strings, rare numbers) and it is the half of a hybrid
retriever that embeddings are worst at.  We use ``rank_bm25`` (the reference
Python implementation) rather than re-deriving the formula.

The tokenizer is the interesting part: see :func:`glassbox.text.tokenize`, which
emits CJK bigrams so BM25 works on Chinese.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from . import text as textmod

try:  # pragma: no cover - import guard
    from rank_bm25 import BM25Okapi

    _HAVE_BM25 = True
except Exception:  # noqa: BLE001
    BM25Okapi = None  # type: ignore[assignment]
    _HAVE_BM25 = False


@dataclass
class SparseHit:
    index: int
    score: float


class SparseIndex:
    """A BM25 index over an ordered list of chunk texts."""

    def __init__(self, corpus: Sequence[str], k1: float = 1.5, b: float = 0.75):
        if not _HAVE_BM25:
            raise RuntimeError("rank_bm25 is not installed — `pip install rank_bm25`")
        self.corpus = list(corpus)
        self.tokens: list[list[str]] = [textmod.tokenize(t) for t in self.corpus]
        # BM25Okapi chokes on a fully empty corpus; keep a 1-token sentinel so
        # `get_scores` stays well-defined and simply returns zeros.
        safe = [t if t else ["\u0000"] for t in self.tokens]
        self.bm25 = BM25Okapi(safe, k1=k1, b=b)
        self._built = True

    def __len__(self) -> int:
        return len(self.corpus)

    def scores(self, query: str) -> list[float]:
        """BM25 score of every chunk against ``query`` (higher is better)."""
        qtok = textmod.tokenize(query)
        if not qtok:
            return [0.0] * len(self.corpus)
        raw = self.bm25.get_scores(qtok)
        return [float(x) for x in raw]

    def search(self, query: str, top_n: int = 30) -> list[SparseHit]:
        sc = self.scores(query)
        order = sorted(range(len(sc)), key=lambda i: (-sc[i], i))
        return [SparseHit(i, sc[i]) for i in order[: max(0, top_n)] if sc[i] > 0.0]

    def vocabulary(self) -> int:
        return len({t for doc in self.tokens for t in doc})
