"""Local generation — a quantised GGUF model on CPU via llama.cpp.

No API keys, no network at inference time.  ``llama-cpp-python`` ships prebuilt
Windows wheels, and a 0.5B Q4_K_M checkpoint is ~400 MB resident, which is what
makes offline generation viable on a 16 GB laptop.

The whole module is *optional*: if llama.cpp or the model file is unavailable,
:attr:`LLM.available` is ``False`` and the pipeline degrades to a grounded
extractive answer instead of failing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, Sequence

from .config import SETTINGS

# hf-xet sometimes fails behind mirrors; the plain HTTP downloader below is
# what we actually use, but set this defensively for huggingface_hub callers.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def model_url(repo: str, filename: str) -> str:
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    return f"{endpoint}/{repo}/resolve/main/{filename}"


def auto_threads(logical: int | None = None) -> int:
    """Pick a thread count for CPU inference.

    A 0.5B checkpoint does **not** scale with core count — it is memory-bandwidth
    bound, so extra threads only add contention.  Measured on a 16-thread Ryzen 7
    with Qwen2.5-0.5B-Q4_K_M (96 tokens, best of two):

        threads   2     4     6     8    12    16
        tok/s    32.3  32.5  25.6  23.6  25.1   8.2

    The naive ``cpu_count() - 1`` lands at 15 threads and ~8 tok/s, i.e. 4x
    slower than the optimum.  We stay at 2-4 threads, which also leaves the rest
    of the machine free (Docker, editors, the retrieval models themselves).
    """
    n = logical if logical is not None else (os.cpu_count() or 4)
    return max(2, min(4, n // 4))


def download_gguf(
    repo: str,
    filename: str,
    dest_dir: Path,
    *,
    progress: bool = False,
) -> Path:
    """Download a GGUF from the configured endpoint.  Idempotent."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    import requests

    url = model_url(repo, filename)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        done = 0
        with open(tmp, "wb") as fh:
            for block in r.iter_content(chunk_size=1 << 20):
                if not block:
                    continue
                fh.write(block)
                done += len(block)
                if progress and total:
                    pct = done * 100.0 / total
                    print(f"\r  {filename}  {pct:5.1f}%  {done/1e6:6.1f}/{total/1e6:.1f} MB", end="")
    if progress:
        print()
    tmp.replace(dest)
    return dest


@dataclass
class LLMConfig:
    repo: str = SETTINGS.llm_repo
    filename: str = SETTINGS.llm_file
    n_ctx: int = SETTINGS.llm_ctx
    n_threads: int = SETTINGS.llm_threads


class LLM:
    """Lazy chat model.  Construction never downloads or loads anything."""

    def __init__(self, cfg: LLMConfig | None = None, models_dir: Path | None = None):
        self.cfg = cfg or LLMConfig()
        self.models_dir = models_dir or SETTINGS.models_dir
        self._model = None
        self._error: str | None = None
        self._tried = False

    # -- availability ------------------------------------------------------
    def model_path(self) -> Path:
        return self.models_dir / self.cfg.filename

    def ensure_model(self, *, progress: bool = False, download: bool = True) -> Path | None:
        p = self.model_path()
        if p.exists() and p.stat().st_size > 0:
            return p
        if not download:
            return None
        try:
            return download_gguf(self.cfg.repo, self.cfg.filename, self.models_dir, progress=progress)
        except Exception as exc:  # noqa: BLE001
            self._error = f"download failed: {exc}"
            return None

    @property
    def available(self) -> bool:
        if not self._tried:
            self._load()
        return self._model is not None

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def name(self) -> str:
        return f"{self.cfg.repo}/{self.cfg.filename}"

    def _load(self) -> None:
        path = self.ensure_model(download=False)
        if path is None:
            # The checkpoint may simply not have finished downloading.  Stay
            # retryable so a model that lands later is picked up by the *same*
            # process — otherwise a server started before the download ends
            # would serve extractive answers forever.  The retry cost is one
            # `Path.exists()` per availability check.
            self._error = f"model not present at {self.model_path()}"
            return
        self._tried = True
        try:
            from llama_cpp import Llama

            threads = self.cfg.n_threads or auto_threads()
            self._model = Llama(
                model_path=str(path),
                n_ctx=self.cfg.n_ctx,
                n_threads=threads,
                n_batch=256,
                verbose=False,
                logits_all=False,
            )
            self._error = None
        except Exception as exc:  # noqa: BLE001
            self._model = None
            self._error = f"llama.cpp load failed: {exc}"

    # -- inference ---------------------------------------------------------
    def chat(
        self,
        messages: Sequence[dict],
        *,
        max_tokens: int = SETTINGS.max_new_tokens,
        temperature: float = SETTINGS.temperature,
        stream: bool = True,
    ) -> Generator[str, None, None]:
        """Yield answer text incrementally.  Yields nothing when unavailable."""
        if not self.available or self._model is None:
            return
        kwargs = dict(
            messages=list(messages),
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            repeat_penalty=1.1,
        )
        if not stream:
            out = self._model.create_chat_completion(**kwargs)
            yield out["choices"][0]["message"]["content"] or ""
            return
        for part in self._model.create_chat_completion(stream=True, **kwargs):
            choices = part.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            piece = delta.get("content")
            if piece:
                yield piece

    def complete(self, messages: Sequence[dict], **kw) -> str:
        return "".join(self.chat(messages, stream=True, **kw))
