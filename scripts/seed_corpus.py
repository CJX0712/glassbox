"""Write the built-in demo corpus out to ``corpus/*.md``.

The engine works without this — :mod:`glassbox.sample_corpus` is used as a
fallback — but materialising the files makes them editable, which is the
point: replace them with your own notes and re-run ``glassbox ingest``.

This script is **idempotent**.  If the target folder already holds indexable
documents it writes nothing, so ``run.sh`` / ``run.bat`` can call it on every
launch without ever clobbering your own notes.  Pass ``--force`` to overwrite
the demo files anyway.

    python scripts/seed_corpus.py [target_dir] [--force]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from glassbox.documents import INDEXABLE_SUFFIXES  # noqa: E402
from glassbox.sample_corpus import SAMPLE_DOCS  # noqa: E402


def has_documents(target: Path) -> bool:
    """True when ``target`` already holds files the engine can index."""
    if not target.is_dir():
        return False
    return any(
        f.is_file() and f.suffix.lower() in INDEXABLE_SUFFIXES
        for f in target.rglob("*")
    )


def main() -> int:
    argv = sys.argv[1:]
    force = "--force" in argv
    positional = [a for a in argv if not a.startswith("-")]
    target = Path(positional[0]) if positional else ROOT / "corpus"

    # Never overwrite a user's own corpus — an empty folder (a fresh clone has
    # one, kept alive by corpus/.gitkeep) must still seed, so test for content
    # rather than for existence.
    if has_documents(target) and not force:
        print(f"[seed] {target.name}/ already holds documents — nothing written")
        return 0

    target.mkdir(parents=True, exist_ok=True)
    for doc_id, title, text in SAMPLE_DOCS:
        path = target / f"{doc_id}.md"
        path.write_text(f"# {title}\n\n{text}\n", encoding="utf-8")
        # A target outside the repo cannot be made relative to ROOT.
        try:
            shown: Path | str = path.relative_to(ROOT)
        except ValueError:
            shown = path
        print(f"  wrote {shown}  ({len(text)} chars)")
    print(f"\n{len(SAMPLE_DOCS)} documents -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
