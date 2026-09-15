#!/usr/bin/env bash
# ---------------------------------------------------------------
#  Glassbox launcher (macOS / Linux)
#    ./run.sh                start the UI
#    ./run.sh --rebuild      re-index corpus/ then start
#    ./run.sh --with-llm     also install llama-cpp-python + the GGUF
# ---------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HERE/.venv/bin/python"

if [ ! -x "$PY" ]; then
  echo "[glassbox] creating virtualenv .venv ..."
  python3 -m venv "$HERE/.venv"
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install -r "$HERE/requirements.txt"
fi

"$PY" -c "import glassbox" 2>/dev/null || "$PY" -m pip install -e "$HERE"

# Idempotent: seeds the demo corpus only when corpus/ holds no documents, so a
# fresh clone (where .gitkeep keeps the folder but leaves it empty) still gets
# the 12 demo documents while your own notes are never overwritten.
"$PY" "$HERE/scripts/seed_corpus.py"

for arg in "$@"; do
  if [ "$arg" = "--with-llm" ]; then
    "$PY" -c "import llama_cpp" 2>/dev/null || {
      echo "[glassbox] installing llama-cpp-python prebuilt wheel ..."
      "$PY" -m pip install llama-cpp-python \
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
        --only-binary llama-cpp-python
    }
    "$PY" -m glassbox download-model
  fi
done

PORT=8765
REBUILD=()
for arg in "$@"; do
  if [ "$arg" = "--rebuild" ]; then REBUILD=(--rebuild); fi
done

echo
echo "[glassbox] http://127.0.0.1:$PORT"
if [ "${#REBUILD[@]}" -gt 0 ]; then
  exec "$PY" -m glassbox serve --port "$PORT" --rebuild
else
  exec "$PY" -m glassbox serve --port "$PORT"
fi
