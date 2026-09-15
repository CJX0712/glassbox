"""Rank fusion.

Sparse (BM25) and dense (embedding) retrieval fail in *different* ways: BM25
misses paraphrases, embeddings miss exact identifiers, rare names and numbers.
Fusing them is what makes hybrid retrieval beat either half alone.

We use **Reciprocal Rank Fusion** (Cormack et al., 2009) rather than score
averaging.  It is the industry default for a blunt reason: BM25 scores are
unbounded and corpus-dependent while cosine similarities live in ``[-1, 1]``,
so any weighted sum needs per-query calibration that RRF simply does not need.
RRF only consumes *ranks*::

    RRF(d) = sum_over_retrievers  w_r / (k + rank_r(d))

with ``k = 60`` the value from the paper.  ``k`` damps the influence of the
very top ranks so a single retriever cannot unilaterally dictate the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence


@dataclass
class Fused:
    """The fused ranking plus the per-retriever evidence that produced it."""

    order: list[int]                       # candidate ids, best first
    scores: dict[int, float]               # id -> fused score
    contributions: dict[int, dict[str, float]] = field(default_factory=dict)
    ranks: dict[int, dict[str, int]] = field(default_factory=dict)


def rrf(
    rankings: Sequence[Sequence[int]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
    names: Sequence[str] | None = None,
) -> Fused:
    """Fuse several best-first rankings of integer ids.

    ``rankings`` is a list of ordered id sequences; ``rankings[0]`` is the
    output of retriever 0.  Duplicate ids inside a single ranking are ignored
    after their first occurrence, which matters because a retriever that
    returns a chunk twice must not get double credit.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must match rankings length")
    names = list(names) if names else [f"r{i}" for i in range(len(rankings))]

    scores: dict[int, float] = {}
    contrib: dict[int, dict[str, float]] = {}
    rank_map: dict[int, dict[str, int]] = {}

    for r, ranking in enumerate(rankings):
        seen: set[int] = set()
        for pos, doc in enumerate(ranking):
            if doc in seen:
                continue
            seen.add(doc)
            add = weights[r] / (k + pos + 1)
            scores[doc] = scores.get(doc, 0.0) + add
            contrib.setdefault(doc, {})[names[r]] = add
            rank_map.setdefault(doc, {})[names[r]] = pos

    order = sorted(scores, key=lambda d: (-scores[d], d))
    return Fused(order=order, scores=scores, contributions=contrib, ranks=rank_map)


def top_n(scores: Sequence[float], n: int, *, descending: bool = True) -> list[int]:
    """Indices of the ``n`` largest (or smallest) scores, best first.

    Ties break by ascending index, which keeps retrieval deterministic — a
    requirement for the reproducibility self-test.
    """
    idx = range(len(scores))
    order = sorted(idx, key=lambda i: ((-scores[i]) if descending else scores[i], i))
    return list(order[: max(0, n)])


def minmax(scores: Sequence[float]) -> list[float]:
    """Scale scores into ``[0, 1]``.  Degenerate inputs map to all-zeros."""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-12:
        return [0.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]
