"""The Glassbox pipeline — ingest, hybrid retrieval, rerank, self-correcting loop.

What makes this a *glass box* rather than a black box is that every stage
records what it did and why:

    query
      ├─ sparse  (BM25)          -> ranked list + scores
      ├─ dense   (embeddings)    -> ranked list + scores
      ├─ fusion  (RRF)           -> fused ranking + per-retriever contribution
      ├─ rerank  (cross-encoder) -> shortlist rescored
      └─ grade   -> answer, or rewrite the query and go round again

The self-correcting step is deliberately **not** an LLM-only trick.  When a
model is available the query is rewritten by the LLM; when it is not, the
pipeline falls back to *pseudo-relevance feedback* (Rocchio-style query
expansion from the current top passage).  Either way the loop works offline and
the trace shows exactly what changed.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

from .config import SETTINGS, Settings, resolve_embed_model, resolve_rerank_model
from .embed import Embedder
from .fusion import rrf
from .llm import LLM
from .rerank import Reranker
from .sparse import SparseIndex
from .store import VectorStore, chunks_from_json, chunks_to_json
from .text import Chunk, chunk_documents, tokenize

# Very small stop-word list: enough to keep coverage meaningful without pulling
# in a corpus-specific vocabulary.
_STOP = {
    "the", "a", "an", "of", "to", "in", "is", "are", "was", "were", "and", "or",
    "for", "on", "at", "by", "with", "as", "that", "this", "it", "be", "from",
    "how", "what", "why", "when", "which", "does", "do", "can", "i", "you",
    "的", "了", "是", "在", "和", "与", "有", "我", "你", "它", "这", "那",
}


@dataclass
class Hit:
    """One candidate, with the evidence every stage produced about it."""

    index: int
    uid: str
    doc_id: str
    title: str
    text: str
    sparse_score: float | None = None
    sparse_rank: int | None = None
    dense_score: float | None = None
    dense_rank: int | None = None
    rrf_score: float | None = None
    rrf_rank: int | None = None
    rerank_score: float | None = None
    rerank_rank: int | None = None
    final_rank: int | None = None


@dataclass
class Trace:
    query: str
    hits: list[Hit]
    stage_sparse: list[dict] = field(default_factory=list)
    stage_dense: list[dict] = field(default_factory=list)
    stage_fused: list[dict] = field(default_factory=list)
    stage_reranked: list[dict] = field(default_factory=list)
    grade: dict = field(default_factory=dict)
    attempts: list[dict] = field(default_factory=list)
    timings_ms: dict = field(default_factory=dict)
    reranker: str | None = None
    backend: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def content_terms(query: str) -> list[str]:
    """Query terms that carry meaning: latin words + CJK bigrams."""
    return [t for t in tokenize(query) if len(t) >= 2 and t not in _STOP]


def coverage(query: str, context: str) -> float:
    """Fraction of the query's content terms that appear in ``context``.

    This is the transparent half of the relevance grade: a cheap, explainable
    signal of whether the retrieved text can even *contain* the answer.
    """
    terms = content_terms(query)
    if not terms:
        return 1.0
    have = set(tokenize(context))
    return sum(1 for t in terms if t in have) / len(terms)


class Glassbox:
    """The engine.  One instance owns the index and all model handles."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
    ):
        self.s = settings or SETTINGS
        self.s.data_dir.mkdir(parents=True, exist_ok=True)

        # Model handles may be injected.  The self-test does this so it can spin
        # up a second, throwaway index without loading a second ONNX session —
        # resident memory is a first-class constraint here.
        self._embedder: Embedder | None = embedder
        self._reranker: Reranker | None = reranker
        self._reranker_resolved = reranker is not None
        self.llm = LLM(models_dir=self.s.models_dir)

        self.chunks: list[Chunk] = []
        self.sparse: SparseIndex | None = None
        self.store: VectorStore | None = None
        self._dense_matrix: np.ndarray | None = None

    # ------------------------------------------------------------------
    # lazy model handles
    # ------------------------------------------------------------------
    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder

    @property
    def reranker(self) -> Reranker:
        if not self._reranker_resolved:
            self._reranker = Reranker(resolve_rerank_model())
            self._reranker_resolved = True
        return self._reranker

    @property
    def dim(self) -> int:
        return self.embedder.dim

    # ------------------------------------------------------------------
    # ingest
    # ------------------------------------------------------------------
    def ingest(
        self,
        docs: Iterable[tuple[str, str, str]],
        *,
        reset: bool = True,
        verbose: bool = False,
    ) -> dict:
        """Chunk, embed and index ``(doc_id, title, text)`` triples."""
        t0 = time.perf_counter()
        docs = list(docs)
        chunks = chunk_documents(
            docs, chunk_chars=self.s.chunk_chars, overlap=self.s.chunk_overlap
        )
        if reset:
            self.chunks = chunks
        else:
            self.chunks.extend(chunks)

        texts = [c.text for c in self.chunks]
        if verbose:
            print(f"  chunked -> {len(chunks)} chunks from {len(docs)} docs")

        t_embed = time.perf_counter()
        vectors = self.embedder.encode_documents(texts)
        embed_ms = (time.perf_counter() - t_embed) * 1000

        t_faiss = time.perf_counter()
        self._dense_matrix = vectors
        self.store = VectorStore(self.embedder.dim, vectors)
        index_ms = (time.perf_counter() - t_faiss) * 1000

        t_sparse = time.perf_counter()
        self.sparse = SparseIndex(texts)
        sparse_ms = (time.perf_counter() - t_sparse) * 1000

        self.save()
        return {
            "documents": len(docs),
            "chunks": len(self.chunks),
            "dim": self.embedder.dim,
            "embed_model": self.embedder.name,
            "vector_backend": self.store.backend if self.store else "",
            "vocabulary": self.sparse.vocabulary() if self.sparse else 0,
            "timings_ms": {
                "embed": round(embed_ms, 1),
                "index": round(index_ms, 1),
                "sparse": round(sparse_ms, 1),
                "total": round((time.perf_counter() - t0) * 1000, 1),
            },
        }

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def save(self) -> None:
        self.s.meta_file.write_text(chunks_to_json(self.chunks), encoding="utf-8")
        if self.store is not None:
            self.store.save(self.s.index_file, self.s.faiss_file)

    def load(self) -> bool:
        """Load a previously built index.  Returns True when one was found.

        The vector count must match the chunk count — a truncated ``.npy`` or a
        half-written index would otherwise load "successfully" and then quietly
        return no results at all.
        """
        if not self.s.meta_file.exists():
            return False
        try:
            self.chunks = chunks_from_json(self.s.meta_file.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - corrupt metadata -> rebuild
            return False
        if not self.chunks:
            return False
        dim = self.embedder.dim
        self.store = VectorStore.load(self.s.index_file, dim)
        if len(self.store) != len(self.chunks):
            self.chunks, self.store = [], None
            return False
        self._dense_matrix = self.store.matrix
        self.sparse = SparseIndex([c.text for c in self.chunks])
        return True

    @property
    def ready(self) -> bool:
        return (
            bool(self.chunks)
            and self.store is not None
            and self.sparse is not None
            and len(self.store) == len(self.chunks)
        )

    # ------------------------------------------------------------------
    # retrieval
    # ------------------------------------------------------------------
    def _dense_search(self, query: str, k: int):
        assert self.store is not None
        qv = self.embedder.encode_query(query)
        return self.store.search(qv, k)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        use_rerank: bool = True,
        fanout: int | None = None,
    ) -> Trace:
        """Run one full retrieval pass and return an evidence-complete trace."""
        if not self.ready:
            raise RuntimeError("index is empty — ingest documents first")

        s = self.s
        top_k = top_k or s.top_k
        fanout = fanout or s.fanout
        t_start = time.perf_counter()
        timings: dict[str, float] = {}

        # --- sparse ----------------------------------------------------
        t = time.perf_counter()
        sparse_scores = self.sparse.scores(query)                        # type: ignore[union-attr]
        sparse_rank_all = sorted(range(len(sparse_scores)), key=lambda i: (-sparse_scores[i], i))
        sparse_top = [i for i in sparse_rank_all[: fanout] if sparse_scores[i] > 0.0]
        sparse_pos = {idx: pos for pos, idx in enumerate(sparse_rank_all)}
        timings["sparse"] = round((time.perf_counter() - t) * 1000, 2)

        # --- dense -----------------------------------------------------
        t = time.perf_counter()
        dense_scores, dense_ids = self._dense_search(query, fanout)
        dense_map = {i: sc for i, sc in zip(dense_ids, dense_scores)}
        dense_pos = {idx: pos for pos, idx in enumerate(dense_ids)}
        timings["dense"] = round((time.perf_counter() - t) * 1000, 2)

        # --- fusion ----------------------------------------------------
        t = time.perf_counter()
        fused = rrf(
            [sparse_top, dense_ids],
            k=s.rrf_k,
            names=["sparse", "dense"],
        )
        fused_order = fused.order[: max(fanout, top_k)]
        timings["fusion"] = round((time.perf_counter() - t) * 1000, 2)

        # --- rerank ----------------------------------------------------
        t = time.perf_counter()
        rr = self.reranker if use_rerank else Reranker(None)
        shortlist = fused_order[: max(s.rerank_top_n * 3, top_k)]
        rr_scores: dict[int, float] = {}
        if rr.enabled and shortlist:
            docs = [self.chunks[i].text for i in shortlist]
            vals = rr.score(query, docs)
            rr_scores = {i: v for i, v in zip(shortlist, vals)}
            reranked_order = sorted(shortlist, key=lambda i: (-rr_scores[i], i))
        else:
            reranked_order = list(shortlist)
        rr_pos = {idx: pos for pos, idx in enumerate(reranked_order)}
        timings["rerank"] = round((time.perf_counter() - t) * 1000, 2)

        # --- assemble --------------------------------------------------
        final_idx = reranked_order[:top_k]
        hits: list[Hit] = []
        for pos, i in enumerate(final_idx):
            c = self.chunks[i]
            hits.append(
                Hit(
                    index=i,
                    uid=c.uid,
                    doc_id=c.doc_id,
                    title=c.doc_title,
                    text=c.text,
                    sparse_score=round(float(sparse_scores[i]), 6),
                    sparse_rank=sparse_pos.get(i),
                    dense_score=round(float(dense_map.get(i, 0.0)), 6),
                    dense_rank=dense_pos.get(i),
                    rrf_score=round(float(fused.scores.get(i, 0.0)), 8),
                    rrf_rank=fused.order.index(i) if i in fused.scores else None,
                    rerank_score=round(rr_scores[i], 6) if i in rr_scores else None,
                    rerank_rank=rr_pos.get(i),
                    final_rank=pos,
                )
            )

        def stage(ids: Sequence[int], scores: dict, cap: int = 12) -> list[dict]:
            out = []
            for pos, i in enumerate(list(ids)[:cap]):
                c = self.chunks[i]
                out.append(
                    {
                        "index": i,
                        "uid": c.uid,
                        "title": c.doc_title,
                        "preview": c.text[:160],
                        "rank": pos,
                        "score": round(float(scores.get(i, 0.0)), 6),
                    }
                )
            return out

        trace = Trace(
            query=query,
            hits=hits,
            stage_sparse=stage(sparse_top, {i: sparse_scores[i] for i in range(len(sparse_scores))}),
            stage_dense=stage(dense_ids, dense_map),
            stage_fused=stage(fused_order, fused.scores),
            stage_reranked=stage(reranked_order, {i: rr_scores.get(i, 0.0) for i in reranked_order}),
            reranker=rr.name if rr.enabled else None,
            backend=self.store.backend if self.store else "",
        )
        timings["total"] = round((time.perf_counter() - t_start) * 1000, 2)
        trace.timings_ms = timings
        return trace

    # ------------------------------------------------------------------
    # relevance grading + query rewriting
    # ------------------------------------------------------------------
    def grade(self, query: str, trace: Trace) -> dict:
        """Decide whether the retrieved evidence is good enough to answer."""
        ctx = "\n".join(h.text for h in trace.hits[: self.s.rerank_top_n])
        cov = coverage(query, ctx)
        top_rr = next((h.rerank_score for h in trace.hits if h.rerank_score is not None), None)
        top_dense = max((h.dense_score or 0.0) for h in trace.hits) if trace.hits else 0.0

        # Two independent signals must agree before we call it sufficient:
        # the evidence must mention the query's terms, and the best passage must
        # not be an obviously weak match.
        floor = self.s.relevance_floor
        enough = cov >= 0.6 and (top_rr is None or top_rr >= floor) and top_dense >= 0.25
        if not trace.hits:
            enough = False
        return {
            "coverage": round(cov, 4),
            "top_rerank": round(top_rr, 4) if top_rr is not None else None,
            "top_dense": round(float(top_dense), 4),
            "threshold_coverage": 0.6,
            "decision": "answer" if enough else "rewrite",
            "reason": (
                "证据已覆盖查询要点" if enough
                else f"证据覆盖不足（{cov:.0%} < 60%），尝试改写查询再检"
            ),
        }

    def rewrite_query(self, query: str, trace: Trace) -> tuple[str, str]:
        """Return ``(new_query, mode)``.  LLM when available, PRF otherwise."""
        if self.llm.available and trace.hits:
            prompt = (
                "Rewrite the user's question into a better search query. "
                "Keep every key entity, add likely synonyms, remove filler. "
                "Reply with the rewritten query only, on one line.\n\n"
                f"Question: {query}\n"
                f"Current best passage: {trace.hits[0].text[:400]}"
            )
            try:
                out = self.llm.complete(
                    [{"role": "user", "content": prompt}], max_tokens=64, temperature=0.0
                )
                out = (out or "").strip().splitlines()[0].strip().strip('"')
                if out and out.lower() != query.lower():
                    return out, "llm"
            except Exception:  # noqa: BLE001 - fall through to PRF
                pass

        # Pseudo-relevance feedback: pull the most distinctive terms out of the
        # current top passage that the query does not already contain.
        have = set(tokenize(query))
        seen: dict[str, int] = {}
        for h in trace.hits[:2]:
            for tok in tokenize(h.text):
                if len(tok) >= 2 and tok not in have:
                    seen[tok] = seen.get(tok, 0) + 1
        expansion = [t for t, _ in sorted(seen.items(), key=lambda kv: (-kv[1], kv[0]))[:4]]
        if not expansion:
            return query, "none"
        return f"{query} {' '.join(expansion)}", "prf"

    def search_with_loop(
        self, query: str, *, top_k: int | None = None, max_rounds: int | None = None
    ) -> tuple[Trace, list[dict]]:
        """Retrieve, grade, and rewrite up to ``max_rounds`` times."""
        rounds = self.s.max_rewrites if max_rounds is None else max_rounds
        attempts: list[dict] = []
        cur = query
        best_trace: Trace | None = None
        best_grade: dict | None = None

        for r in range(rounds + 1):
            trace = self.retrieve(cur, top_k=top_k)
            g = self.grade(cur, trace)
            attempts.append(
                {
                    "round": r,
                    "query": cur,
                    "coverage": g["coverage"],
                    "top_dense": g["top_dense"],
                    "top_rerank": g["top_rerank"],
                    "decision": g["decision"],
                    "reason": g["reason"],
                    "mode": "original" if r == 0 else attempts[-1]["mode"] if attempts else "original",
                }
            )
            if best_grade is None or g["coverage"] > best_grade["coverage"]:
                best_trace, best_grade = trace, g
            if g["decision"] == "answer" or r == rounds:
                break
            new_q, mode = self.rewrite_query(cur, trace)
            if new_q == cur:
                break
            attempts[-1]["mode"] = mode
            attempts.append({"round": r, "rewritten_to": new_q, "mode": mode})
            cur = new_q

        assert best_trace is not None and best_grade is not None
        best_trace.grade = best_grade
        best_trace.attempts = attempts
        return best_trace, attempts

    # ------------------------------------------------------------------
    # answering
    # ------------------------------------------------------------------
    def build_messages(self, query: str, trace: Trace, k: int | None = None) -> list[dict]:
        k = k or self.s.rerank_top_n
        blocks = []
        for n, h in enumerate(trace.hits[:k], start=1):
            blocks.append(f"[{n}] ({h.title})\n{h.text}")
        context = "\n\n".join(blocks) if blocks else "(no context retrieved)"
        system = (
            "You are Glassbox, a fully offline retrieval-augmented assistant. "
            "Answer using ONLY the numbered context. Cite sources as [1], [2]. "
            "If the context does not contain the answer, say so plainly instead "
            "of guessing. Be concise."
        )
        user = f"Context:\n{context}\n\nQuestion: {query}"
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def extractive_answer(self, query: str, trace: Trace, k: int | None = None) -> str:
        """Grounded fallback used when no local model is loaded."""
        k = k or min(3, len(trace.hits))
        lines = ["未加载本地生成模型，以下为**基于检索的直接摘录**（grounded extract）：", ""]
        for n, h in enumerate(trace.hits[:k], start=1):
            snippet = h.text.strip()
            if len(snippet) > 320:
                snippet = snippet[:320].rstrip() + "…"
            lines.append(f"[{n}] **{h.title}** — {snippet}")
        return "\n".join(lines)

    def answer(self, query: str, *, stream: bool = True, top_k: int | None = None) -> Iterator[dict]:
        """Yield events: ``trace`` then ``token``* then ``done``."""
        t0 = time.perf_counter()
        trace, attempts = self.search_with_loop(query, top_k=top_k)
        mode = "generative" if self.llm.available else "extractive"
        yield {
            "type": "trace",
            "data": trace.to_dict(),
            "mode": mode,
            "llm": self.llm.name if self.llm.available else None,
        }

        if mode == "generative":
            messages = self.build_messages(query, trace)
            for piece in self.llm.chat(messages, stream=True):
                yield {"type": "token", "text": piece}
        else:
            text = self.extractive_answer(query, trace)
            for piece in _chunks(text, 24):
                yield {"type": "token", "text": piece}

        yield {
            "type": "done",
            "mode": mode,
            "citations": [
                {"n": n, "uid": h.uid, "title": h.title}
                for n, h in enumerate(trace.hits[: self.s.rerank_top_n], start=1)
            ],
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    def answer_sync(self, query: str, **kw) -> dict:
        """Non-streaming convenience wrapper (used by the CLI and self-test)."""
        text, trace, meta = "", None, {}
        for ev in self.answer(query, stream=False, **kw):
            if ev["type"] == "trace":
                trace = ev["data"]
                meta["mode"] = ev["mode"]
            elif ev["type"] == "token":
                text += ev["text"]
        return {"answer": text, "trace": trace, **meta}

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        return {
            "ready": self.ready,
            "chunks": len(self.chunks),
            "documents": len({c.doc_id for c in self.chunks}),
            "dim": self.dim if self._embedder else None,
            "embed_model": self.embedder.name,
            "reranker": self.reranker.name if self.reranker.enabled else None,
            "vector_backend": self.store.backend if self.store else None,
            "llm": self.llm.name if self.llm.available else None,
            "llm_error": self.llm.error,
            "data_dir": str(self.s.data_dir),
        }


def _chunks(text: str, size: int) -> Iterator[str]:
    for i in range(0, len(text), size):
        yield text[i : i + size]
