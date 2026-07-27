#!/usr/bin/env python3
"""Pre-flight audit before launching the ladder (guide section 7.1).

Checks, per ladder slot found in .env (managed block or manual quads):
  1. BASE_URL is set and every pod answers /v1/models: 401 without key,
     200 with VLLM_API_KEY, and the served model id matches *_AGENT_MODEL_NAME.
     A slot may be served by several pods: *_AGENT_MODEL_BASE_URLS (plural,
     comma-separated, what 3_sync_runpod_pods_env.py writes) is read first,
     falling back to the singular *_AGENT_MODEL_BASE_URL; every URL is checked.
  2. The three protective flags are set to the required values
     (AGENT_MODEL_MAX_TOKENS_CAP=8192, AGENT_MODEL_STREAMING=1,
     AGENT_MODEL_DISABLE_THINKING=1).
  3. Optional smoke completion (default ON, --no-smoke to skip): one tiny
     chat call per pod; PASS = non-empty content and no reasoning_content
     (proves non-thinking mode actually took effect server-side).
  4. Free disk space for results (warn below --min-disk-gb, default 10).

Reads .env itself (no `source` needed). Exit code 0 = all PASS, 1 = any FAIL.

Usage:
  python scripts/alignmentgraph-isd-bench/01_audit_preflight.py
  python scripts/alignmentgraph-isd-bench/01_audit_preflight.py --no-smoke
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SLOTS = ["qwen08b", "qwen2b", "qwen4b", "qwen9b"]
REQUIRED_FLAGS = {
    "AGENT_MODEL_MAX_TOKENS_CAP": "8192",
    "AGENT_MODEL_STREAMING": "1",
    "AGENT_MODEL_DISABLE_THINKING": "1",
}

_fail = 0


def report(ok: bool, label: str, detail: str = "") -> None:
    global _fail
    mark = "PASS" if ok else "FAIL"
    if not ok:
        _fail += 1
    print(f"  [{mark}] {label}" + (f" — {detail}" if detail else ""))


def load_env_file(path: Path) -> None:
    """Best-effort .env loader: KEY=V / export KEY=V; os.environ wins."""
    if not path.exists():
        print(f"WARNING: {path} not found; relying on the current environment",
              file=sys.stderr)
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def slot_base_urls(upper: str) -> list[str]:
    """Base URLs for a slot, plural first (what the sync script writes).

    A slot may be served by several pods of the same model; the launchers fold
    <SLOT>_AGENT_MODEL_BASE_URLS into the singular and the agents round-robin
    across them (shared/llm/config.py::resolve_base_url). Accept both spellings
    here and audit every pod, not just the first.
    """
    raw = (os.environ.get(f"{upper}_AGENT_MODEL_BASE_URLS")
           or os.environ.get(f"{upper}_AGENT_MODEL_BASE_URL")
           or "")
    return [u.strip() for u in raw.split(",") if u.strip()]


# RunPod's proxy (Cloudflare) 403s the default Python-urllib User-Agent;
# any curl-like UA passes through to vLLM (which then answers 401/200).
UA = {"User-Agent": "curl/8.4.0 (isd-agent-benchmark audit)"}


def http_get(url: str, key: str | None, timeout: int = 15):
    headers = dict(UA)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, ""
    except Exception as e:
        return None, str(e)


def http_post_json(url: str, key: str, payload: dict, timeout: int = 120):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={**UA, "Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, {}
    except Exception as e:
        return None, {"error": str(e)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-smoke", action="store_true",
                        help="Skip the per-pod smoke completion call")
    parser.add_argument("--min-disk-gb", type=float, default=10.0)
    args = parser.parse_args()

    load_env_file(REPO_ROOT / ".env")
    vllm_key = os.environ.get("VLLM_API_KEY")

    print("== Pre-flight audit (guide 7.1) ==")

    # 1) pods alive + auth + model match
    any_slot = False
    for slot in SLOTS:
        upper = slot.upper()
        base_urls = slot_base_urls(upper)
        name = os.environ.get(f"{upper}_AGENT_MODEL_NAME", "")
        print(f"\n-- {slot} ({name or 'no NAME'})")
        if not base_urls:
            report(False, "BASE_URL(S) set",
                   "missing — run scripts/3_sync_runpod_pods_env.py")
            continue
        any_slot = True
        for idx, base_url in enumerate(base_urls, 1):
            # Only tag the pod index when a slot is served by several pods, so
            # the single-pod output stays exactly as it was.
            tag = f"pod {idx}/{len(base_urls)}: " if len(base_urls) > 1 else ""
            if tag:
                print(f"   {base_url}")
            status_nokey, _ = http_get(f"{base_url}/models", None)
            hint = " (404 = pod stopped/not running?)" if status_nokey == 404 else ""
            report(status_nokey == 401, f"{tag}auth required (401 without key)",
                   f"got {status_nokey}{hint}")
            if not vllm_key:
                report(False, "VLLM_API_KEY present", "not set in .env")
                break
            status, body = http_get(f"{base_url}/models", vllm_key)
            report(status == 200, f"{tag}reachable with key (200)", f"got {status}")
            if status == 200:
                try:
                    served = [m["id"] for m in json.loads(body).get("data", [])]
                except (json.JSONDecodeError, KeyError):
                    served = []
                report(name in served, f"{tag}served model matches NAME",
                       f"server: {served}")
                if not args.no_smoke:
                    # Replicate ladder client conditions: non-thinking is a
                    # CLIENT-side request flag (AGENT_MODEL_DISABLE_THINKING=1 ->
                    # chat_template_kwargs), not a server default -- without it
                    # the model burns the tiny budget inside <think> and content
                    # comes back empty.
                    st, doc = http_post_json(
                        f"{base_url}/chat/completions", vllm_key,
                        {"model": name, "max_tokens": 32,
                         "chat_template_kwargs": {"enable_thinking": False},
                         "messages": [{"role": "user", "content": "Say OK"}]},
                    )
                    msg = (doc.get("choices") or [{}])[0].get("message", {})
                    content_ok = bool((msg.get("content") or "").strip())
                    no_think = not msg.get("reasoning_content")
                    report(st == 200 and content_ok, f"{tag}smoke: content non-empty",
                           f"status={st}")
                    report(no_think, f"{tag}smoke: no reasoning_content (non-thinking)",
                           "reasoning_content present!" if not no_think else "")

    if not any_slot:
        print("\nNo slot has a BASE_URL — nothing to check against.")

    # 2) protective flags
    print("\n-- protective flags")
    for key, want in REQUIRED_FLAGS.items():
        got = os.environ.get(key)
        report(got == want, f"{key}={want}", f"got {got!r}")

    # 3) disk
    print("\n-- disk")
    free_gb = shutil.disk_usage(REPO_ROOT).free / 1e9
    report(free_gb >= args.min_disk_gb,
           f"free disk >= {args.min_disk_gb:.0f}GB", f"{free_gb:.1f}GB free")

    print(f"\n== {'ALL PASS' if _fail == 0 else f'{_fail} FAIL(s)'} ==")
    return 0 if _fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
