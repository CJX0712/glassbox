"""Glassbox command line.

    python -m glassbox serve                      # start the web UI
    python -m glassbox ask "为什么用 RRF"          # one-shot question
    python -m glassbox retrieve "RRF"             # retrieval only, full trace
    python -m glassbox ingest ./my-notes          # index a folder or a file
    python -m glassbox selftest                   # invariant suite
    python -m glassbox download-model             # fetch the local GGUF
    python -m glassbox models                     # show available models
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import documents as docmod
from . import sample_corpus
from .config import SETTINGS, describe
from .pipeline import Glassbox


def _load_engine(args: argparse.Namespace, *, rebuild: bool = False) -> Glassbox:
    eng = Glassbox()
    if rebuild:
        src = Path(args.path) if getattr(args, "path", None) else eng.s.corpus_dir
        docs = docmod.load_corpus_dir(src) if src.is_dir() else (
            [t for t in [docmod.load_file(src)] if t] if src.exists() else []
        )
        if not docs:
            docs = sample_corpus.triples()
        info = eng.ingest(docs, reset=True, verbose=True)
        print(f"  indexed {info['chunks']} chunks / {info['documents']} docs in "
              f"{info['timings_ms']['total']:.0f} ms")
    elif not eng.load():
        docs = docmod.load_corpus_dir(eng.s.corpus_dir) or sample_corpus.triples()
        info = eng.ingest(docs, reset=True)
        print(f"  built index: {info['chunks']} chunks / {info['documents']} docs")
    return eng


# Ports to try when the requested one cannot be bound.  On Windows an
# administrator-reserved range (Hyper-V / WinNAT) can make a perfectly ordinary
# port like 8000 fail with WinError 10013, which looks like a bug in the app
# but is really the OS refusing the bind.
FALLBACK_PORTS = (8765, 8801, 9000, 9080, 8712, 8788, 8517)


def pick_port(host: str, port: int) -> int:
    """Return the requested port if it binds, otherwise the first that does."""
    import socket

    for candidate in (port, *[p for p in FALLBACK_PORTS if p != port]):
        with socket.socket() as sock:
            try:
                sock.bind((host, candidate))
                return candidate
            except OSError:
                continue
    return port


# --------------------------------------------------------------------------
def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    if args.rebuild:
        _load_engine(args, rebuild=True)
    port = pick_port(args.host, args.port)
    if port != args.port:
        print(f"  port {args.port} is not bindable here — using {port}")
    print(f"Glassbox UI  ->  http://{args.host}:{port}")
    uvicorn.run("glassbox.api:app", host=args.host, port=port, log_level="warning")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    eng = _load_engine(args)
    result = eng.answer_sync(args.question, top_k=args.top_k)
    trace = result.get("trace") or {}
    grade = trace.get("grade") or {}
    print()
    print(f"Q  {args.question}")
    print(f"   模式={result.get('mode')}  覆盖率={grade.get('coverage')}  判定={grade.get('decision')}")
    print("-" * 76)
    print(result["answer"])
    print("-" * 76)
    for n, h in enumerate((trace.get("hits") or [])[:5], start=1):
        print(f"[{n}] {h['title']}  ({h['uid']})")
    return 0


def cmd_retrieve(args: argparse.Namespace) -> int:
    eng = _load_engine(args)
    trace, attempts = eng.search_with_loop(args.question, top_k=args.top_k)
    print()
    print(f"Q  {args.question}")
    for a in attempts:
        if "rewritten_to" in a:
            print(f"   -> rewrite({a['mode']}): {a['rewritten_to']}")
        else:
            print(f"   R{a['round']} coverage={a['coverage']:.2f} decision={a['decision']}")
    print()
    hdr = f"{'#':<3}{'uid':<18}{'BM25':>8}{'dense':>9}{'RRF':>10}{'rerank':>10}  title"
    print(hdr)
    print("-" * len(hdr))
    for n, h in enumerate(trace.hits, start=1):
        def fmt(v, w):
            return f"{v:>{w}.4f}" if isinstance(v, (int, float)) else f"{'-':>{w}}"
        print(
            f"{n:<3}{h.uid:<18}{fmt(h.sparse_rank, 8)}{fmt(h.dense_rank, 9)}"
            f"{fmt(h.rrf_score, 10)}{fmt(h.rerank_score, 10)}  {h.title}"
        )
    print()
    print("timings(ms):", trace.timings_ms)
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    _load_engine(args, rebuild=True)
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    from . import selftest as st

    return st.main()


def cmd_download_model(args: argparse.Namespace) -> int:
    from .llm import LLM, model_url

    llm = LLM()
    print(f"  endpoint : {model_url(llm.cfg.repo, llm.cfg.filename).rsplit('/resolve', 1)[0]}")
    print(f"  target   : {llm.model_path()}")
    path = llm.ensure_model(progress=True, download=True)
    if path is None:
        print(f"  FAILED: {llm.error}")
        return 1
    print(f"  ok: {path.name}  {path.stat().st_size/1e6:.1f} MB")
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    info = describe()
    for k, v in info.items():
        if isinstance(v, list):
            print(f"{k}:")
            for item in v:
                print(f"    {item}")
        else:
            print(f"{k}: {v}")
    return 0


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="glassbox", description="Offline glass-box RAG engine")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="start the web UI")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--rebuild", action="store_true", help="re-index from corpus/ first")
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser("ask", help="ask one question")
    s.add_argument("question")
    s.add_argument("--top-k", type=int, default=SETTINGS.top_k)
    s.set_defaults(func=cmd_ask)

    s = sub.add_parser("retrieve", help="retrieval only, print the trace")
    s.add_argument("question")
    s.add_argument("--top-k", type=int, default=SETTINGS.top_k)
    s.set_defaults(func=cmd_retrieve)

    s = sub.add_parser("ingest", help="build the index from a path")
    s.add_argument("path", nargs="?", default=None)
    s.set_defaults(func=cmd_ingest)

    s = sub.add_parser("selftest", help="run the invariant suite")
    s.set_defaults(func=cmd_selftest)

    s = sub.add_parser("download-model", help="download the local GGUF")
    s.set_defaults(func=cmd_download_model)

    s = sub.add_parser("models", help="list available models")
    s.set_defaults(func=cmd_models)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
