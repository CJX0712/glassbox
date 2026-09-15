"""FastAPI application — the HTTP surface of Glassbox.

Endpoints
---------
``GET  /``                 the single-file UI
``GET  /api/bootstrap``    engine state + resolved models (builds the index once)
``POST /api/ingest``       index new documents (JSON) or an uploaded file
``POST /api/reset``        rebuild the index from the ``corpus/`` folder
``GET  /api/retrieve``     retrieval only — returns the full glass-box trace
``POST /api/ask``          retrieval + generation, streamed as SSE
``GET  /api/selftest``     run the invariant suite
``GET  /api/stats``        index statistics
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import documents as docmod
from . import sample_corpus
from . import selftest as selftest_mod
from .config import SETTINGS, describe
from .pipeline import Glassbox

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "static"

app = FastAPI(title="Glassbox", version="1.0.0")
_lock = threading.Lock()
_engine: Glassbox | None = None
_state: dict[str, Any] = {"ingest": None, "error": None, "source": None}


def engine() -> Glassbox:
    """Return the process-wide engine, building or loading the index once."""
    global _engine
    if _engine is not None and _engine.ready:
        return _engine
    with _lock:
        if _engine is not None and _engine.ready:
            return _engine
        eng = Glassbox(SETTINGS)
        try:
            if not eng.load():
                # Decide the source by inspecting the folder, not by comparing
                # object identity: ``triples()`` returns a fresh list, so an
                # ``is`` check would report "corpus/" even for the builtin set.
                docs = docmod.load_corpus_dir(eng.s.corpus_dir)
                _state["source"] = "corpus/" if docs else "builtin"
                _state["ingest"] = eng.ingest(docs or sample_corpus.triples(), reset=True)
            _state["error"] = None
        except Exception as exc:  # noqa: BLE001
            _state["error"] = str(exc)
            raise
        _engine = eng
        return eng


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


# --------------------------------------------------------------------------
# pages
# --------------------------------------------------------------------------
@app.get("/")
def index() -> FileResponse:
    page = STATIC_DIR / "index.html"
    if not page.exists():
        raise HTTPException(500, "static/index.html is missing")
    return FileResponse(page)


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# --------------------------------------------------------------------------
# api
# --------------------------------------------------------------------------
@app.get("/api/bootstrap")
def bootstrap() -> JSONResponse:
    try:
        eng = engine()
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(exc), "models": describe()}, status_code=500)
    return JSONResponse(
        {
            "ok": True,
            "stats": eng.stats(),
            "ingest": _state.get("ingest"),
            "models": describe(),
            "settings": {
                "chunk_chars": eng.s.chunk_chars,
                "chunk_overlap": eng.s.chunk_overlap,
                "rrf_k": eng.s.rrf_k,
                "rerank_weight": eng.s.rerank_weight,
                "top_k": eng.s.top_k,
                "fanout": eng.s.fanout,
                "max_rewrites": eng.s.max_rewrites,
            },
        }
    )


@app.get("/api/stats")
def stats() -> JSONResponse:
    return JSONResponse(engine().stats())


@app.post("/api/reset")
def reset() -> JSONResponse:
    global _engine
    with _lock:
        eng = Glassbox(SETTINGS)
        docs = docmod.load_corpus_dir(eng.s.corpus_dir)
        _state["source"] = "corpus/" if docs else "builtin"
        info = eng.ingest(docs or sample_corpus.triples(), reset=True)
        _state["ingest"] = info
        _engine = eng
    return JSONResponse({"ok": True, "ingest": info, "stats": eng.stats()})


@app.post("/api/ingest")
async def ingest_json(payload: dict) -> JSONResponse:
    eng = engine()
    triples = docmod.as_triples(payload.get("documents") or [])
    if not triples:
        raise HTTPException(400, "no usable documents in payload")
    info = eng.ingest(triples, reset=False)
    return JSONResponse({"ok": True, "added": len(triples), "ingest": info})


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> JSONResponse:
    eng = engine()
    tmp = eng.s.data_dir / "_uploads"
    tmp.mkdir(parents=True, exist_ok=True)
    dest = tmp / (file.filename or "upload.md")
    dest.write_bytes(await file.read())
    triple = docmod.load_file(dest, doc_id=(file.filename or "upload").rsplit(".", 1)[0])
    if not triple:
        raise HTTPException(400, f"unsupported or empty file: {file.filename}")
    info = eng.ingest([triple], reset=False)
    return JSONResponse({"ok": True, "ingested": triple[0], "ingest": info})


@app.get("/api/retrieve")
def retrieve(q: str = Query(..., min_length=1), top_k: int = 6) -> JSONResponse:
    eng = engine()
    trace, attempts = eng.search_with_loop(q, top_k=top_k)
    return JSONResponse({"ok": True, "trace": trace.to_dict(), "attempts": attempts})


@app.post("/api/ask")
def ask(payload: dict) -> StreamingResponse:
    query = (payload.get("query") or "").strip()
    if not query:
        raise HTTPException(400, "query is required")
    top_k = int(payload.get("top_k") or SETTINGS.top_k)
    eng = engine()

    def gen():
        try:
            for event in eng.answer(query, top_k=top_k):
                yield _sse(event)
        except Exception as exc:  # noqa: BLE001 - surface errors to the UI
            yield _sse({"type": "error", "message": str(exc)})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/selftest")
def selftest() -> JSONResponse:
    eng = engine()
    checks = selftest_mod.run_all(eng)
    return JSONResponse(
        {
            "ok": all(c.passed for c in checks),
            "passed": sum(1 for c in checks if c.passed),
            "total": len(checks),
            "checks": [c.__dict__ for c in checks],
        }
    )


@app.get("/api/models")
def models() -> JSONResponse:
    return JSONResponse(describe())
