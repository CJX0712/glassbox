"""Document loading: a corpus folder, individual files, and PDFs.

The engine only ever sees ``(doc_id, title, text)`` triples, so this module is
the single place that knows how to turn files on disk into those triples.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".rst", ".text"}
PDF_SUFFIXES = {".pdf"}

#: Every suffix the engine can index — the single source of truth for
#: "does this folder already hold usable documents?".
INDEXABLE_SUFFIXES = TEXT_SUFFIXES | PDF_SUFFIXES


def _title_of(path: Path, text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip() or path.stem
        if line:
            return line[:80]
    return path.stem


def load_file(path: Path, doc_id: str | None = None) -> tuple[str, str, str] | None:
    """Read one file into a triple.  Returns ``None`` for unsupported types."""
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        text = path.read_text(encoding="utf-8", errors="replace")
    elif suffix in PDF_SUFFIXES:
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception:  # noqa: BLE001 - a broken PDF must not kill ingest
            return None
    else:
        return None
    text = text.strip()
    if not text:
        return None
    return (doc_id or path.stem, _title_of(path, text), text)


def load_corpus_dir(root: Path) -> list[tuple[str, str, str]]:
    """Load every supported file under ``root`` (recursive, sorted)."""
    if not root.exists():
        return []
    out: list[tuple[str, str, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in INDEXABLE_SUFFIXES:
            continue
        triple = load_file(path, doc_id=str(path.relative_to(root)).replace("\\", "/"))
        if triple:
            out.append(triple)
    return out


def as_triples(payload: Iterable[dict]) -> list[tuple[str, str, str]]:
    """Coerce an API payload into triples, skipping malformed entries."""
    out: list[tuple[str, str, str]] = []
    for i, item in enumerate(payload):
        if not isinstance(item, dict):
            continue
        text = (item.get("text") or "").strip()
        if not text:
            continue
        out.append(
            (
                str(item.get("id") or f"doc{i}"),
                str(item.get("title") or item.get("id") or f"doc{i}"),
                text,
            )
        )
    return out
