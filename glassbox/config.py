"""Glassbox configuration: paths, model registry and runtime knobs.

Everything Glassbox runs with is decided in this one module.

Model selection is *adaptive* rather than hard-coded: the embedder and the
cross-encoder reranker are resolved at runtime from whatever the installed
``fastembed`` build actually ships.  We prefer multilingual checkpoints so the
engine handles Chinese and English out of the box, and we fall back gracefully
when a model is unavailable instead of crashing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# Hugging Face endpoint
# --------------------------------------------------------------------------
# huggingface.co is not reachable from this host, but hf-mirror.com is
# (verified: plain HTTP 200 + ranged object download 206).  Set the endpoint
# *before* huggingface_hub is imported anywhere.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("GLASSBOX_DATA", str(ROOT / ".glassbox")))
CORPUS_DIR = ROOT / "corpus"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Settings:
    """Runtime settings.  All of them are overridable through the environment."""

    data_dir: Path = DATA_DIR
    corpus_dir: Path = CORPUS_DIR

    # --- chunking ---------------------------------------------------------
    chunk_chars: int = _env_int("GLASSBOX_CHUNK_CHARS", 700)
    chunk_overlap: int = _env_int("GLASSBOX_CHUNK_OVERLAP", 180)

    # --- retrieval --------------------------------------------------------
    rrf_k: int = _env_int("GLASSBOX_RRF_K", 60)
    top_k: int = _env_int("GLASSBOX_TOPK", 8)
    rerank_top_n: int = _env_int("GLASSBOX_RERANK_TOPN", 6)
    # How many candidates each first-stage retriever hands to the fusion step.
    fanout: int = _env_int("GLASSBOX_FANOUT", 30)

    # --- agentic loop -----------------------------------------------------
    # A chunk is "relevant" when its cross-encoder score clears this.  The
    # grade decides whether the pipeline answers, or rewrites and retries.
    relevance_floor: float = float(os.environ.get("GLASSBOX_RELEVANCE_FLOOR", 0.0))
    max_rewrites: int = _env_int("GLASSBOX_MAX_REWRITES", 2)

    # --- generation -------------------------------------------------------
    llm_repo: str = os.environ.get(
        "GLASSBOX_LLM_REPO", "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
    )
    llm_file: str = os.environ.get(
        "GLASSBOX_LLM_FILE", "qwen2.5-0.5b-instruct-q4_k_m.gguf"
    )
    llm_ctx: int = _env_int("GLASSBOX_LLM_CTX", 4096)
    llm_threads: int = _env_int("GLASSBOX_LLM_THREADS", 0)  # 0 -> auto
    max_new_tokens: int = _env_int("GLASSBOX_MAX_NEW_TOKENS", 512)
    temperature: float = float(os.environ.get("GLASSBOX_TEMPERATURE", 0.2))

    @property
    def index_file(self) -> Path:
        return self.data_dir / "index.npz"

    @property
    def meta_file(self) -> Path:
        return self.data_dir / "chunks.json"

    @property
    def faiss_file(self) -> Path:
        return self.data_dir / "vectors.faiss"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"


SETTINGS = Settings()


# --------------------------------------------------------------------------
# Model registry
# --------------------------------------------------------------------------
# Ordered best-first.  The resolver returns the first entry the installed
# fastembed build actually supports.
EMBED_PREFERENCE: tuple[str, ...] = (
    # multilingual / Chinese-capable first — Glassbox ships a bilingual corpus
    "BAAI/bge-small-zh-v1.5",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "intfloat/multilingual-e5-small",
    "intfloat/multilingual-e5-base",
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-base-en-v1.5",
)

RERANK_PREFERENCE: tuple[str, ...] = (
    "jinaai/jina-reranker-v1-turbo-en",
    "Xenova/ms-marco-MiniLM-L-6-v2",
    "Xenova/ms-marco-MiniLM-L-12-v2",
    "BAAI/bge-reranker-base",
)


def _supported(kind: str) -> dict[str, dict[str, Any]]:
    """Return ``{model_name: metadata}`` for fastembed models of ``kind``."""
    out: dict[str, dict[str, Any]] = {}
    try:
        if kind == "embed":
            from fastembed import TextEmbedding

            for m in TextEmbedding.list_supported_models():
                out[m["model"]] = m
        else:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            for m in TextCrossEncoder.list_supported_models():
                out[m["model"]] = m
    except Exception:  # noqa: BLE001 - registry must never break import
        return {}
    return out


def resolve_embed_model() -> tuple[str, int]:
    """Pick the best available embedding model.  Returns ``(name, dim)``."""
    avail = _supported("embed")
    override = os.environ.get("GLASSBOX_EMBED_MODEL")
    if override and override in avail:
        return override, int(avail[override].get("dim", 0))
    for name in EMBED_PREFERENCE:
        if name in avail:
            return name, int(avail[name].get("dim", 0))
    # last resort: first supported model with a known dimension
    for name, meta in avail.items():
        if meta.get("dim"):
            return name, int(meta["dim"])
    raise RuntimeError(
        "no fastembed embedding model available — did `pip install fastembed` succeed?"
    )


def resolve_rerank_model() -> str | None:
    """Pick the best available cross-encoder.  ``None`` disables reranking."""
    if os.environ.get("GLASSBOX_DISABLE_RERANK", "").lower() in {"1", "true", "yes"}:
        return None
    avail = _supported("rerank")
    override = os.environ.get("GLASSBOX_RERANK_MODEL")
    if override:
        return override if override in avail else None
    for name in RERANK_PREFERENCE:
        if name in avail:
            return name
    # fall back to anything rerank-capable
    for name in avail:
        return name
    return None


def describe() -> dict[str, Any]:
    """A JSON-friendly description of the resolved environment."""
    out: dict[str, Any] = {"hf_endpoint": os.environ.get("HF_ENDPOINT")}
    for kind in ("embed", "rerank"):
        models = _supported(kind)
        out[f"{kind}_available"] = sorted(models)
        out[f"{kind}_count"] = len(models)
    try:
        name, dim = resolve_embed_model()
        out["embed_selected"], out["embed_dim"] = name, dim
    except Exception as exc:  # noqa: BLE001
        out["embed_selected"], out["embed_error"] = None, str(exc)
    out["rerank_selected"] = resolve_rerank_model()
    return out
