"""Text normalisation, sentence-aware chunking and a CJK-friendly tokenizer.

Two things in here are worth calling out because they are where naive RAG
implementations quietly lose quality:

1. **Chunking.**  Splitting on a fixed character count slices sentences in
   half, which is exactly where the answer usually lives.  :func:`chunk_text`
   packs *whole sentences* into a window and carries an overlap so a fact that
   straddles a boundary still appears intact in at least one chunk.

2. **Tokenisation for BM25.**  The textbook ``text.split()`` tokenizer is
   useless on Chinese, where there are no spaces.  :func:`tokenize` emits
   lower-cased latin/digit words *and* CJK character bigrams, which is the
   standard cheap approximation of Chinese word segmentation and needs no
   extra dependency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

# --------------------------------------------------------------------------
# Character classes
# --------------------------------------------------------------------------
# CJK ideographs, kana and hangul.  Anything in here gets bigram treatment.
_CJK_RANGE = (
    r"\u3040-\u30ff"      # kana
    r"\u3400-\u4dbf"      # CJK ext A
    r"\u4e00-\u9fff"      # CJK unified
    r"\uf900-\ufaff"      # compatibility ideographs
    r"\uac00-\ud7af"      # hangul
)
_CJK_RE = re.compile(f"[{_CJK_RANGE}]")
_LATIN_RE = re.compile(r"[A-Za-z0-9]+(?:['\u2019-][A-Za-z0-9]+)*")
_WS_RE = re.compile(r"[ \t\u3000]+")

# Sentence terminators: latin and CJK punctuation, plus hard newlines.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?\u3002\uff01\uff1f\uff1b;\n])\s*")


def normalize(text: str) -> str:
    """Collapse runs of horizontal whitespace, keep paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RE.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    """Split into sentences, keeping the terminator.  Never returns empties."""
    parts = _SENT_SPLIT_RE.split(normalize(text))
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if p:
            out.append(p)
    return out


def tokenize(text: str) -> list[str]:
    """Tokenise for BM25: latin words + CJK unigrams/bigrams + digits.

    Chinese has no word delimiters, so a whitespace tokenizer would emit one
    token per sentence and BM25 would degenerate into exact text matching.
    Character bigrams recover most of the signal for a fraction of the cost of
    a real segmenter: ``玻璃盒`` -> ``玻 玻璃 璃盒 盒``.
    """
    tokens: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if _CJK_RE.match(ch):
            run = []
            while i < n and _CJK_RE.match(text[i]):
                run.append(text[i])
                i += 1
            tokens.extend(run)                       # unigrams
            tokens.extend(run[j] + run[j + 1] for j in range(len(run) - 1))
            continue
        m = _LATIN_RE.match(text, i)
        if m:
            tokens.append(m.group(0).lower())
            i = m.end()
            continue
        i += 1
    return tokens


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
@dataclass
class Chunk:
    """One retrievable unit of text."""

    doc_id: str
    doc_title: str
    chunk_id: int
    text: str
    start: int          # character offset of this chunk inside the document
    sentences: int      # how many sentences it holds

    @property
    def uid(self) -> str:
        return f"{self.doc_id}#{self.chunk_id}"


def chunk_text(
    text: str,
    *,
    chunk_chars: int = 700,
    overlap: int = 180,
    doc_id: str = "doc",
    doc_title: str = "",
) -> list[Chunk]:
    """Pack whole sentences into ~``chunk_chars`` windows with ``overlap``.

    Invariants (asserted by ``glassbox.selftest``):

    * no sentence is dropped — every sentence of the source appears in at
      least one chunk;
    * chunk boundaries never fall inside a sentence.

    The overlap is expressed in *sentences*, derived from ``overlap``
    characters, so a chunk always repeats whole trailing sentences rather than
    a ragged tail of characters.
    """
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive")
    overlap = max(0, min(overlap, chunk_chars - 1))

    sents = split_sentences(text)
    if not sents:
        return []

    # Character offset of every sentence, so we can report chunk positions.
    offsets: list[int] = []
    cursor = 0
    norm = normalize(text)
    for s in sents:
        idx = norm.find(s, cursor)
        if idx < 0:
            idx = cursor
        offsets.append(idx)
        cursor = idx + len(s)

    chunks: list[Chunk] = []
    i = 0
    cid = 0
    while i < len(sents):
        buf: list[str] = []
        size = 0
        j = i
        while j < len(sents):
            add = len(sents[j]) + (1 if buf else 0)
            if buf and size + add > chunk_chars:
                break
            buf.append(sents[j])
            size += add
            j += 1
        chunks.append(
            Chunk(
                doc_id=doc_id,
                doc_title=doc_title or doc_id,
                chunk_id=cid,
                text=" ".join(buf),
                start=offsets[i],
                sentences=len(buf),
            )
        )
        cid += 1
        if j >= len(sents):
            break
        # Step back far enough to repeat roughly `overlap` characters of tail,
        # but never re-emit the whole chunk and never fail to advance:
        #   * floor at chunk_start + 1  -> at least one sentence of overlap
        #   * cap below j               -> the window always moves forward
        # A single-sentence chunk therefore carries no overlap; that is the one
        # case where overlap is impossible without looping forever.
        chunk_start = j - len(buf)
        nxt = j
        back = 0
        while nxt > 0 and back < overlap:
            nxt -= 1
            back += len(sents[nxt]) + 1
        i = max(nxt, chunk_start + 1)
        if i >= j:
            i = j
    return chunks


def chunk_documents(
    docs: Iterable[tuple[str, str, str]],
    *,
    chunk_chars: int = 700,
    overlap: int = 180,
) -> list[Chunk]:
    """Chunk ``(doc_id, title, text)`` triples."""
    out: list[Chunk] = []
    for doc_id, title, text in docs:
        out.extend(
            chunk_text(
                text,
                chunk_chars=chunk_chars,
                overlap=overlap,
                doc_id=doc_id,
                doc_title=title,
            )
        )
    return out


def coverage(chunks: Sequence[Chunk], text: str) -> float:
    """Fraction of the source's sentences that survive chunking.

    Used by the self-test to prove chunking is loss-free.  A return value of
    ``1.0`` means every sentence is present in at least one chunk.
    """
    sents = split_sentences(text)
    if not sents:
        return 1.0
    joined = "\n".join(c.text for c in chunks)
    hits = sum(1 for s in sents if s in joined)
    return hits / len(sents)
