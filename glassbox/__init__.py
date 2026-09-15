"""Glassbox — an offline, glass-box retrieval-augmented generation engine.

The name is the thesis: every stage of retrieval is measurable and *shown* —
BM25 ranks, dense ranks, the RRF fusion, the cross-encoder rerank and the
relevance grade that decides whether to answer or to rewrite the query and go
round again.

Quick start::

    from glassbox import Glassbox

    eng = Glassbox()
    eng.ingest([("d1", "Title", "some text ...")])
    result = eng.answer_sync("your question")
    print(result["answer"])
"""

from .config import SETTINGS, Settings, resolve_embed_model, resolve_rerank_model
from .pipeline import Glassbox, Trace

__version__ = "1.0.0"
__all__ = [
    "Glassbox",
    "Trace",
    "Settings",
    "SETTINGS",
    "resolve_embed_model",
    "resolve_rerank_model",
    "__version__",
]
