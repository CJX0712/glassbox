"""The invariant suite — Glassbox's proof that it is not hallucinating structure.

Every check here is a property that must hold *by construction* and that can be
verified without trusting any single implementation:

* chunking is loss-free and overlap is real;
* embeddings really are unit vectors, so cosine == inner product;
* FAISS and numpy — two independent implementations — return the same
  neighbours for the same query;
* BM25 is monotone in term frequency, which is the defining property of the
  Okapi formula;
* RRF obeys its own closed form and de-duplicates repeated ids;
* the trace is complete and the pipeline is deterministic.

Run it with ``python -m glassbox.selftest`` or ``GET /selftest``.
"""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .config import Settings, resolve_embed_model, resolve_rerank_model
from .fusion import rrf
from .llm import auto_threads
from .sparse import SparseIndex
from .text import chunk_text, coverage as text_coverage, split_sentences, tokenize

if TYPE_CHECKING:  # pragma: no cover
    from .pipeline import Glassbox


@dataclass
class Check:
    group: str
    name: str
    passed: bool
    detail: str


# --------------------------------------------------------------------------
# Deterministic mini-corpus used by the suite
# --------------------------------------------------------------------------
TEST_DOCS: list[tuple[str, str, str]] = [
    (
        "fusion",
        "Reciprocal Rank Fusion",
        "Reciprocal Rank Fusion combines several ranked lists into one. "
        "Each document receives the sum of one over k plus its rank, across every retriever. "
        "The constant k damps the influence of the very top positions. "
        "Fusion needs only ranks, so it is immune to incompatible score scales. "
        "That robustness is why RRF is the default in hybrid search stacks.",
    ),
    (
        "bm25",
        "Okapi BM25",
        "BM25 is a bag-of-words ranking function. "
        "It scores a document with the sum over query terms of inverse document frequency "
        "times a saturated term frequency factor. "
        "The saturation parameter b normalises for document length. "
        "BM25 excels at exact identifiers and rare strings that embeddings blur away.",
    ),
    (
        "dense",
        "Dense retrieval",
        "Dense retrieval maps queries and passages into one vector space. "
        "Cosine similarity then ranks passages by semantic proximity. "
        "Because meaning is encoded rather than spelled, dense retrieval survives paraphrases. "
        "Its weakness is the opposite of BM25: rare tokens and long numbers are compressed away.",
    ),
    (
        "rerank",
        "Cross-encoder reranking",
        "A cross-encoder reads the query and a passage together and emits a relevance score. "
        "That joint encoding is far more accurate than comparing two independent vectors. "
        "It is also far slower, so it is applied only to a shortlist produced by a cheap first stage. "
        "The two stage retrieve then rerank pattern is standard in production search.",
    ),
    (
        "chunk",
        "Chunking strategy",
        "Splitting documents on a fixed character count cuts sentences in half. "
        "Sentence aware chunking packs whole sentences and carries an overlap. "
        "The overlap guarantees that a fact spanning a boundary still appears intact once. "
        "Chunk size trades recall against precision: small chunks localise, large chunks contextualise.",
    ),
    (
        "term",
        "Term frequency example",
        "A rare word repeated many times tests the saturation behaviour of scoring. "
        "The word saturation appears here. Saturation saturation saturation appears again. "
        "Documents with more occurrences of a term should never score lower than documents with fewer. "
        "This single monotonicity property distinguishes BM25 from raw count matching.",
    ),
]


def _tmp_engine(base: "Glassbox") -> "Glassbox":
    """A second engine over a throwaway index that *shares* the base's models.

    Sharing matters: instantiating a second ONNX embedder would double resident
    memory for no benefit, and this project deliberately runs on tight RAM.
    """
    from .pipeline import Glassbox

    tmp = Path(tempfile.mkdtemp(prefix="glassbox-selftest-"))
    s = Settings(data_dir=tmp)
    eng = Glassbox(s, embedder=base.embedder, reranker=base.reranker)
    return eng


def run_all(engine: "Glassbox") -> list[Check]:
    out: list[Check] = []

    def add(group: str, name: str, passed: bool, detail: str = "") -> None:
        out.append(Check(group, name, bool(passed), detail))

    # ==================================================================
    # 1. chunking
    # ==================================================================
    src = TEST_DOCS[0][2] + " " + TEST_DOCS[1][2]
    chunks = chunk_text(src, chunk_chars=240, overlap=90, doc_id="d", doc_title="d")
    cov = text_coverage(chunks, src)
    add("chunking", "无句子丢失（覆盖率 = 1.0）", cov == 1.0, f"coverage={cov:.4f}")
    add("chunking", "产生多个 chunk", len(chunks) >= 3, f"n={len(chunks)}")

    over = [c for c in chunks if len(c.text) > 240 + 200]
    add("chunking", "chunk 长度受控", not over, f"oversized={len(over)}")

    # Overlap is guaranteed for any chunk holding more than one sentence; a
    # single-sentence chunk cannot overlap without the window failing to move.
    multi_pairs = multi_shared = 0
    for a, b in zip(chunks, chunks[1:]):
        if a.sentences < 2:
            continue
        multi_pairs += 1
        if set(split_sentences(a.text)) & set(split_sentences(b.text)):
            multi_shared += 1
    add(
        "chunking",
        "多句 chunk 与后一块必有重叠",
        multi_pairs > 0 and multi_shared == multi_pairs,
        f"{multi_shared}/{multi_pairs}",
    )

    # ==================================================================
    # 2. embeddings
    # ==================================================================
    emb = engine.embedder
    sample = [d[2] for d in TEST_DOCS[:4]]
    mat = emb.encode_documents(sample)
    norms = np.linalg.norm(mat, axis=1)
    add(
        "embedding",
        "向量已 L2 归一（‖v‖ = 1）",
        bool(np.allclose(norms, 1.0, atol=1e-5)),
        f"max|‖v‖-1|={np.max(np.abs(norms-1)):.2e}",
    )
    add("embedding", "维度与声明一致", mat.shape[1] == emb.dim, f"dim={emb.dim}")

    mat2 = emb.encode_documents(sample)
    add(
        "embedding",
        "编码确定性（逐元素一致）",
        bool(np.array_equal(mat, mat2)),
        f"max|Δ|={np.max(np.abs(mat-mat2)):.2e}",
    )

    qv = emb.encode_query("reciprocal rank fusion")
    sims = emb.similarities(qv, mat)
    add(
        "embedding",
        "余弦相似度 ∈ [-1, 1]",
        bool(np.all(sims <= 1.0 + 1e-6) and np.all(sims >= -1.0 - 1e-6)),
        f"range=[{sims.min():.4f}, {sims.max():.4f}]",
    )
    # cosine == inner product, exactly, because rows are unit vectors
    add(
        "embedding",
        "余弦 ≡ 内积（归一化推论）",
        bool(np.allclose(sims, mat @ qv, atol=1e-6)),
        f"max|Δ|={np.max(np.abs(sims - mat @ qv)):.2e}",
    )

    # ==================================================================
    # 3. sparse / BM25
    # ==================================================================
    # Two things have to be true for this to be a *real* test:
    #
    # * every document is exactly 6 tokens long, so ``b`` length-normalisation
    #   cancels and term frequency is the only variable;
    # * the query term is rare (df=3 of 7) so its IDF stays positive.
    #
    # The previous corpus failed the second condition: both "saturation" and
    # "filler" sat in 3 of 4 documents, giving them negative IDF — and because
    # rank_bm25 replaces negative IDF with ``0.25 * average_idf``, and the
    # average here was exactly 0.0, every score collapsed to 0.0.  The
    # monotonicity assertion then passed vacuously.  Keep the term rare.
    corpus = [
        "saturation alpha alpha alpha alpha alpha",
        "saturation saturation alpha alpha alpha alpha",
        "saturation saturation saturation alpha alpha alpha",
        "bramble thorn hedge thicket briar nettle",
        "quokka wombat bilby numbat dingo wallaby",
        "tundra taiga steppe savanna prairie pampas",
        "kelp coral atoll lagoon reef shoal",
    ]
    sp = SparseIndex(corpus)
    sc = sp.scores("saturation")
    add(
        "sparse",
        "BM25 对词频严格递增",
        sc[0] < sc[1] < sc[2],
        f"scores={[round(x, 4) for x in sc[:3]]}",
    )
    # Independent closed form of Okapi BM25 on an equal-length corpus:
    #   score(tf) = idf * tf*(k1+1) / (tf + k1*(1-b+b*dl/avgdl)),  dl == avgdl
    k1, b = 1.5, 0.75
    idf = math.log(len(corpus) - 3 + 0.5) - math.log(3 + 0.5)
    expected = [idf * tf * (k1 + 1) / (tf + k1) for tf in (1, 2, 3)]
    delta = max(abs(a - e) for a, e in zip(sc[:3], expected))
    add(
        "sparse",
        "BM25 命中解析闭合形式",
        delta < 1e-9,
        f"max|Δ|={delta:.2e}  idf={idf:.6f}",
    )
    add("sparse", "无关文档得分为 0", sc[6] == 0.0, f"score={sc[6]:.4f}")
    add("sparse", "非负得分", all(x >= 0 for x in sc), f"min={min(sc):.4f}")

    toks = tokenize("玻璃盒 glassbox 2026")
    add(
        "sparse",
        "混合分词（CJK 二元 + 拉丁词）",
        "glassbox" in toks and "玻璃" in toks and "2026" in toks,
        f"tokens={toks}",
    )

    # ==================================================================
    # 4. fusion / RRF
    # ==================================================================
    a = [10, 20, 30, 40]
    b = [10, 30, 20, 50]
    f = rrf([a, b], k=60, names=["sparse", "dense"])
    expected_10 = 1 / 61 + 1 / 61
    add(
        "fusion",
        "RRF 闭合形式 1/(k+rank+1)",
        abs(f.scores[10] - expected_10) < 1e-12,
        f"{f.scores[10]:.8f} vs {expected_10:.8f}",
    )
    add(
        "fusion",
        "双榜第一必居融合榜首",
        f.order[0] == 10,
        f"order={f.order[:4]}",
    )
    dup = rrf([[7, 7, 7]], k=60)
    add(
        "fusion",
        "重复 id 只计一次",
        abs(dup.scores[7] - 1 / 61) < 1e-12,
        f"{dup.scores[7]:.8f}",
    )
    add(
        "fusion",
        "融合顺序的分数单调不增",
        all(f.scores[f.order[i]] >= f.scores[f.order[i + 1]] - 1e-12 for i in range(len(f.order) - 1)),
        f"scores={[round(f.scores[i],5) for i in f.order[:4]]}",
    )

    # ==================================================================
    # 5. pipeline end to end (on the throwaway index)
    # ==================================================================
    small = _tmp_engine(engine)
    info = small.ingest(TEST_DOCS, reset=True)
    add("index", "入库产生 chunk", info["chunks"] >= len(TEST_DOCS), f"chunks={info['chunks']}")
    add(
        "index",
        "向量库后端就绪",
        small.store is not None and len(small.store) == info["chunks"],
        f"backend={info['vector_backend']} n={len(small.store) if small.store else 0}",
    )

    # FAISS vs brute force — two independent implementations must agree
    qvec = small.embedder.encode_query("cross encoder reranking shortlist")
    k = min(5, len(small.store))
    s_faiss, i_faiss = small.store.search(qvec, k)
    s_bf, i_bf = small.store.brute_force(qvec, k)
    same_ids = list(i_faiss) == list(i_bf)
    same_scores = bool(np.allclose(s_faiss, s_bf, atol=1e-5))
    add("index", "FAISS 与暴力搜索返回同一批邻居", same_ids, f"faiss={list(i_faiss)} bf={list(i_bf)}")
    add("index", "FAISS 与暴力搜索得分一致", same_scores, f"max|Δ|={np.max(np.abs(np.array(s_faiss)-np.array(s_bf))):.2e}")

    # A query that actually shares vocabulary with the corpus, so every stage
    # has something to contribute. (A purely semantic query would leave BM25
    # legitimately empty — that case is checked separately below.)
    q_en = "cross encoder reranking shortlist"
    tr = small.retrieve(q_en, top_k=4)
    add("retrieval", "最终结果数 = top_k", len(tr.hits) == 4, f"n={len(tr.hits)}")
    add(
        "retrieval",
        "final_rank 连续从 0 开始",
        [h.final_rank for h in tr.hits] == list(range(len(tr.hits))),
        f"ranks={[h.final_rank for h in tr.hits]}",
    )
    add(
        "retrieval",
        "每个命中都留有阶段证据",
        all(h.sparse_rank is not None or h.dense_rank is not None for h in tr.hits),
        "ok" if all(h.sparse_rank is not None or h.dense_rank is not None for h in tr.hits) else "missing",
    )
    add(
        "retrieval",
        "证据链四个阶段俱全",
        all([tr.stage_sparse, tr.stage_dense, tr.stage_fused, tr.stage_reranked]),
        f"sizes={len(tr.stage_sparse)},{len(tr.stage_dense)},{len(tr.stage_fused)},{len(tr.stage_reranked)}",
    )
    # paraphrase with no surface overlap: dense is what carries it
    tr2 = small.retrieve("combining two ranked lists without comparable scores", top_k=3)
    top_titles = {h.doc_id for h in tr2.hits[:2]}
    add("retrieval", "语义查询命中正确文档", "fusion" in top_titles, f"top={sorted(top_titles)}")
    # cross-lingual: a Chinese query over an English corpus. BM25 correctly
    # finds nothing (no shared tokens); the multilingual embedder is the only
    # reason this works at all — and the pipeline must not error out.
    trx = small.retrieve("交叉编码器重排", top_k=3)
    add(
        "retrieval",
        "跨语言查询由稠密阶段兜底",
        bool(trx.hits) and all(h.dense_rank is not None for h in trx.hits),
        f"sparse={len(trx.stage_sparse)} dense={len(trx.stage_dense)} top={[h.doc_id for h in trx.hits[:2]]}",
    )

    tr3 = small.retrieve(q_en, top_k=4)
    same = [h.index for h in tr.hits] == [h.index for h in tr3.hits]
    add("retrieval", "检索确定性（同查询同结果）", same, f"{[h.index for h in tr.hits]}")

    trace, attempts = small.search_with_loop("那是什么", top_k=3)
    add("loop", "自我纠错循环产生尝试记录", len(attempts) >= 1, f"attempts={len(attempts)}")
    add(
        "loop",
        "grade 含覆盖率与决策",
        "coverage" in trace.grade and trace.grade["decision"] in {"answer", "rewrite"},
        f"decision={trace.grade['decision']} cov={trace.grade['coverage']}",
    )
    add(
        "loop",
        "覆盖率 ∈ [0,1]",
        0.0 <= trace.grade["coverage"] <= 1.0,
        f"coverage={trace.grade['coverage']}",
    )

    # ==================================================================
    # 6. reranker (if present)
    # ==================================================================
    if engine.reranker.enabled:
        docs = [d[2] for d in TEST_DOCS[:5]]
        r1 = engine.reranker.score("cross encoder reranking", docs)
        r2 = engine.reranker.score("cross encoder reranking", docs)
        add("rerank", "重排器可运行", len(r1) == len(docs), f"n={len(r1)}")
        add("rerank", "重排确定性", r1 == r2, f"max|Δ|={max(abs(x-y) for x,y in zip(r1,r2)):.2e}")
        best = int(np.argmax(r1))
        add(
            "rerank",
            "重排把最相关段落排到第一",
            TEST_DOCS[best][0] == "rerank",
            f"top={TEST_DOCS[best][0]}",
        )
    else:
        # Graceful degradation is the right behaviour on a laptop that is
        # offline, and a trap in CI: with no reranker the group below has
        # nothing to say, so a failed checkpoint download would *shrink* the
        # suite instead of failing it — a green run that never exercised the
        # code under test.  GLASSBOX_REQUIRE_RERANK turns that silence red.
        required = os.environ.get("GLASSBOX_REQUIRE_RERANK", "").lower() in {"1", "true", "yes"}
        add(
            "rerank",
            "重排器必须启用（GLASSBOX_REQUIRE_RERANK）" if required else "重排器（未启用，已优雅降级）",
            not required,
            f"model={resolve_rerank_model()}",
        )

    # The reranker is a signal, not a veto.  Inject a hostile cross-encoder
    # that scores the shortlist in exactly the reverse order and assert the
    # first-stage fusion's winner still comes out on top.  This is a regression
    # guard for a real bug: the final order used to *be* the reranked order, so
    # an English-only reranker silently demoted correct Chinese results
    # (10/12 vs 11/12 top-1 on the bundled corpus).  With the second-stage
    # weight below 1.0 the arithmetic guarantees the guard cannot flake:
    # winner 1/(k+1) + 0.5/(k+n) beats runner-up 1/(k+n) + 0.5/(k+1).
    class _AdversarialReranker:
        name = "adversarial"
        enabled = True

        def score(self, query: str, documents) -> list[float]:
            # best-first ordering of the worst possible kind: a full reversal
            return [float(i) for i in range(len(documents))]

    probe = "cross encoder reranking shortlist"
    plain = engine.retrieve(probe, top_k=3, use_rerank=False)
    saved = (engine._reranker, engine._reranker_resolved)
    engine._reranker, engine._reranker_resolved = _AdversarialReranker(), True
    try:
        hostile = engine.retrieve(probe, top_k=3, use_rerank=True)
    finally:
        engine._reranker, engine._reranker_resolved = saved
    add(
        "rerank",
        "对抗性重排无法推翻融合榜首",
        hostile.hits[0].index == plain.hits[0].index,
        f"fused#{plain.hits[0].index} -> final#{hostile.hits[0].index} (w={engine.s.rerank_weight})",
    )
    add(
        "rerank",
        "重排仍作为独立信号留证",
        hostile.reranker == "adversarial" and bool(hostile.stage_reranked),
        f"stage_reranked={len(hostile.stage_reranked)} entries",
    )

    # ==================================================================
    # 7. generation runtime
    # ==================================================================
    # A small quantised model is memory-bandwidth bound: oversubscribing it with
    # one thread per logical core is 4x *slower* than 4 threads (measured, see
    # ``llm.auto_threads``).  Guard the band so nobody "optimises" it back.
    picks = [(n, auto_threads(n)) for n in (2, 4, 8, 16, 32, 64)]
    add(
        "generation",
        "自动线程数落在 2-4 区间",
        all(2 <= t <= 4 for _, t in picks),
        " ".join(f"{n}c->{t}t" for n, t in picks),
    )
    add(
        "generation",
        "自动线程数不超过逻辑核数",
        all(t <= n for n, t in picks),
        "ok",
    )
    add(
        "generation",
        "LLM 可加载或已优雅降级",
        engine.llm.available or bool(engine.llm.error),
        (engine.llm.name if engine.llm.available else f"extractive fallback: {engine.llm.error}")[:70],
    )
    ans = engine.extractive_answer("什么是重排", engine.search_with_loop("什么是重排", top_k=3)[0])
    add("generation", "无模型时仍有 grounded 摘录答案", len(ans.strip()) > 40, f"{len(ans)} chars")

    return out


def format_report(checks: list[Check]) -> str:
    lines: list[str] = []
    passed = sum(1 for c in checks if c.passed)
    groups: dict[str, list[Check]] = {}
    for c in checks:
        groups.setdefault(c.group, []).append(c)
    for g, items in groups.items():
        lines.append(f"\n[{g}]")
        for c in items:
            mark = "PASS" if c.passed else "FAIL"
            lines.append(f"  {mark}  {c.name}" + (f"   ({c.detail})" if c.detail else ""))
    lines.append("")
    lines.append(f"{passed}/{len(checks)} checks passed")
    lines.append("ALL GREEN" if passed == len(checks) else "SOME CHECKS FAILED")
    return "\n".join(lines)


def main() -> int:
    """CLI entry point.

    Deliberately builds its index in a **throwaway directory**: running the
    self-test must never clobber the real ``.glassbox`` index that the server
    loads on startup.
    """
    from .pipeline import Glassbox

    tmp = Path(tempfile.mkdtemp(prefix="glassbox-selftest-cli-"))
    eng = Glassbox(Settings(data_dir=tmp))
    eng.ingest(TEST_DOCS, reset=True)
    checks = run_all(eng)
    print(format_report(checks))
    return 0 if all(c.passed for c in checks) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
