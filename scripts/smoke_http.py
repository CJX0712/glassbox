"""End-to-end HTTP smoke test — run it against a live Glassbox server.

    python -m glassbox serve &            # or run.bat / run.sh
    python scripts/smoke_http.py [base_url]

Checks the whole surface: bootstrap, retrieval trace shape, the SSE answer
stream, and the invariant suite.  Exits non-zero on the first failure so it can
be wired into CI.
"""

from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765"
FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


def get(path: str) -> dict:
    with urllib.request.urlopen(BASE + path, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode("utf-8"))


def post_stream(path: str, payload: dict) -> list[dict]:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    events: list[dict] = []
    with urllib.request.urlopen(req, timeout=600) as r:
        buf = ""
        for raw in r:
            buf += raw.decode("utf-8")
            while "\n\n" in buf:
                chunk, buf = buf.split("\n\n", 1)
                if chunk.startswith("data: "):
                    events.append(json.loads(chunk[6:]))
    return events


def main() -> int:
    print(f"Glassbox HTTP smoke -> {BASE}\n")

    print("[bootstrap]")
    b = get("/api/bootstrap")
    check("bootstrap ok", b.get("ok") is True, json.dumps(b.get("stats", {}))[:90])
    stats = b.get("stats", {})
    check("index has chunks", (stats.get("chunks") or 0) > 0, f"chunks={stats.get('chunks')}")
    check("embedding model resolved", bool(stats.get("embed_model")), str(stats.get("embed_model")))
    check("vector backend reported", bool(stats.get("vector_backend")), str(stats.get("vector_backend")))

    print("\n[retrieve]")
    tr = get("/api/retrieve?q=" + urllib.parse.quote("为什么混合检索要用 RRF") + "&top_k=5")
    check("retrieve ok", tr.get("ok") is True)
    trace = tr.get("trace", {})
    hits = trace.get("hits", [])
    check("hits non-empty", len(hits) > 0, f"n={len(hits)}")
    check("final ranks contiguous", [h["final_rank"] for h in hits] == list(range(len(hits))))
    check(
        "four evidence stages present",
        all(trace.get(k) for k in ("stage_sparse", "stage_dense", "stage_fused", "stage_reranked")),
    )
    check("grade present", "coverage" in (trace.get("grade") or {}),
          json.dumps(trace.get("grade", {}), ensure_ascii=False)[:90])
    rrf_hit = next((h for h in hits if h.get("rrf_score")), None)
    check("fused evidence recorded", rrf_hit is not None)

    print("\n[ask / SSE]")
    events = post_stream("/api/ask", {"query": "chunk overlap 解决了什么问题", "top_k": 5})
    kinds = [e.get("type") for e in events]
    check("trace event first", kinds[:1] == ["trace"], str(kinds[:3]))
    check("streamed tokens", kinds.count("token") > 0, f"tokens={kinds.count('token')}")
    done = next((e for e in events if e.get("type") == "done"), None)
    check("done event present", done is not None)
    check("citations returned", bool(done and done.get("citations")), f"n={len(done.get('citations', [])) if done else 0}")
    text = "".join(e.get("text", "") for e in events if e.get("type") == "token")
    check("answer text non-empty", len(text.strip()) > 20, f"{len(text)} chars")
    if done:
        print(f"        mode={done.get('mode')}  elapsed={done.get('elapsed_ms')}ms")

    print("\n[selftest]")
    st = get("/api/selftest")
    check("all invariants pass", st.get("ok") is True, f"{st.get('passed')}/{st.get('total')}")
    for c in st.get("checks", []):
        if not c.get("passed"):
            print(f"        FAILED: {c.get('group')} / {c.get('name')}  {c.get('detail')}")

    print()
    print(f"{'ALL GREEN' if not FAILS else str(len(FAILS)) + ' FAILURES: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
