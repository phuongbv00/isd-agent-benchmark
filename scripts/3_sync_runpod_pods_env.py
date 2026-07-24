"""Sync ladder env vars from RunPod Pods straight into .env.

The ladder pods are created MANUALLY in the RunPod console from the public
vLLM template (one template for all 4 sizes; only the model -- the first
positional token of the start command -- differs per pod; see
../docs/benchmark_guides.md, section 4, for the template link and console steps).
This script does the remaining mechanical part: list the live pods, match
each one to a ladder slot (by start command / pod name), confirm the served
model against the pod's own /v1/models, and REWRITE the managed block at the
end of the repo's .env file (between the >>>/<<< markers below) with the
current env-quads -- so `source .env` (which 5_run_ladder.sh already does)
picks everything up. Re-running replaces the block, never duplicates it;
nothing outside the markers is touched.

It only ever READS from the RunPod API (GET /pods). Creating, stopping and
deleting pods stays a console operation.

Requires: RUNPOD_API_KEY in .env (https://www.runpod.io/console/user/settings) --
loaded automatically via python-dotenv, no need to `source .env` first.

Usage:
  python scripts/3_sync_runpod_pods_env.py            # sync pods -> .env block
  python scripts/3_sync_runpod_pods_env.py --dry-run  # print the block, write nothing
  source .env
  ./scripts/alignmentgraph-isd-bench/5_run_ladder.sh

The managed block contains, per slot, the benchmark env-quad:

  QWEN2B_AGENT_MODEL_PROVIDER=runpod-vllm
  QWEN2B_AGENT_MODEL_BASE_URL=https://<pod_id>-8000.proxy.runpod.net/v1
  QWEN2B_AGENT_MODEL_NAME=Qwen/Qwen3.5-2B
  QWEN2B_AGENT_MODEL_API_KEY_ENVS=VLLM_API_KEY

plus the three protective flags every ladder run must have (see
../docs/benchmark_guides.md, section 4.4, for the incidents that motivated them):

  AGENT_MODEL_MAX_TOKENS_CAP=8096
  AGENT_MODEL_STREAMING=1
  AGENT_MODEL_DISABLE_THINKING=1

VLLM_API_KEY itself must already be set in .env -- it is the Bearer
token the template's vLLM server was started with, not the RunPod API key.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

REST_BASE = "https://rest.runpod.io/v1"
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
BLOCK_BEGIN = "# >>> qwen-ladder pods (managed by scripts/3_sync_runpod_pods_env.py) >>>"
BLOCK_END = "# <<< qwen-ladder pods <<<"

try:
    from dotenv import load_dotenv
    load_dotenv(ENV_PATH)
except ImportError:
    pass  # Skip if dotenv is not available

# Ladder slot <- size marker, matched case-insensitively against the pod's
# vLLM start command (the model is its first positional token -- the template
# has no shell, so env vars can't carry it), falling back to the pod name.
SLOT_PATTERNS = [
    ("qwen08b", re.compile(r"0[._]8b", re.I)),
    ("qwen2b", re.compile(r"(?<![\d.])2b", re.I)),
    ("qwen4b", re.compile(r"(?<![\d.])4b", re.I)),
    ("qwen9b", re.compile(r"(?<![\d.])9b", re.I)),
]

PROTECTIVE_FLAGS = [
    "AGENT_MODEL_MAX_TOKENS_CAP=8096",
    "AGENT_MODEL_STREAMING=1",
    "AGENT_MODEL_DISABLE_THINKING=1",
]


def _rest(path: str) -> object:
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("ERROR: RUNPOD_API_KEY is not set (see .env).", file=sys.stderr)
        sys.exit(1)
    req = urllib.request.Request(
        REST_BASE + path, headers={"Authorization": f"Bearer {api_key}"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"ERROR: GET {path} -> HTTP {e.code}: {e.read().decode()[:300]}", file=sys.stderr)
        sys.exit(1)


def _pod_model_hint(pod: dict) -> str:
    """Best-effort HF model id from the pod record.

    The template model is the FIRST POSITIONAL token of the start command
    (RunPod appends dockerStartCmd as plain Docker CMD -- no shell, no env
    expansion), so look there first; fall back to a MODEL_NAME env if some
    pod sets one anyway, then to the pod display name for slot matching.
    """
    cmd = pod.get("dockerStartCmd") or pod.get("dockerArgs") or ""
    if isinstance(cmd, list):
        cmd = " ".join(str(c) for c in cmd)
    for token in str(cmd).split():
        if not token.startswith("-") and "/" in token:  # e.g. Qwen/Qwen3.5-2B
            return token
    env = pod.get("env") or {}
    if isinstance(env, dict) and env.get("MODEL_NAME"):
        return str(env["MODEL_NAME"])
    return ""


def _match_slot(*candidates: str) -> str | None:
    for text in candidates:
        if not text:
            continue
        for slot, pattern in SLOT_PATTERNS:
            if pattern.search(text):
                return slot
    return None


def _served_model(base_url: str) -> str | None:
    """Ask the pod's vLLM server which model it actually serves (needs
    VLLM_API_KEY). Authoritative over anything parsed from the pod record --
    a wrong AGENT_MODEL_NAME makes the OpenAI-compatible server 404 every
    request. Returns None if unreachable (pod booting, key unset, ...)."""
    vllm_key = os.environ.get("VLLM_API_KEY")
    if not vllm_key:
        return None
    req = urllib.request.Request(
        base_url + "/models",
        headers={"Authorization": f"Bearer {vllm_key}",
                 # RunPod proxy 403s the default Python-urllib UA
                 "User-Agent": "curl/8.4.0 (isd-agent-benchmark sync)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        models = [m.get("id") for m in data.get("data", []) if m.get("id")]
        return models[0] if models else None
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8000,
                        help="vLLM port exposed via the RunPod proxy (default: 8000)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the pod->slot mapping without writing files")
    args = parser.parse_args()

    data = _rest("/pods")
    pods = data if isinstance(data, list) else data.get("pods", [])

    slots: dict[str, dict] = {}
    print(f"{len(pods)} pod(s) on the account:")
    for pod in pods:
        pod_id = pod.get("id", "?")
        name = pod.get("name", "")
        status = pod.get("desiredStatus", "?")
        model_hint = _pod_model_hint(pod)
        slot = _match_slot(model_hint, name)
        base_url = f"https://{pod_id}-{args.port}.proxy.runpod.net/v1"

        # Authoritative check: what does the vLLM server itself serve?
        served = _served_model(base_url) if slot else None
        model_name = served or model_hint
        verified = "live" if served else "unverified"

        tag = slot or "-"
        print(f"  [{tag:8s}] {pod_id}  {status:8s}  model={model_name or '?'} ({verified})  name={name}")
        if slot is None:
            continue
        if served and model_hint and served != model_hint:
            print(f"           NOTE: pod record says {model_hint!r} but server serves "
                  f"{served!r}; using the server's id", file=sys.stderr)
        if not served:
            print(f"           WARNING: could not confirm via {base_url}/models "
                  f"(pod booting? VLLM_API_KEY unset?) -- using {model_name!r} unverified",
                  file=sys.stderr)
        if slot in slots:
            print(f"           WARNING: multiple pods match {slot}; keeping the first "
                  f"({slots[slot]['pod_id']}), ignoring {pod_id}", file=sys.stderr)
            continue
        if status != "RUNNING":
            print(f"           WARNING: {slot} pod is {status}, not RUNNING", file=sys.stderr)
        slots[slot] = {
            "pod_id": pod_id,
            "pod_name": name,
            "status": status,
            "hf_model": model_name,
            "model_verified_live": bool(served),
            "base_url": base_url,
            "gpu_type": (pod.get("machine") or {}).get("gpuTypeId") or pod.get("gpuTypeId"),
        }

    missing = [s for s, _ in SLOT_PATTERNS if s not in slots]
    if missing:
        print(f"\nWARNING: no pod matched slot(s): {', '.join(missing)} "
              f"(deploy them in the console first; see ../docs/benchmark_guides.md, section 4)",
              file=sys.stderr)
    if not slots:
        print("ERROR: no ladder pods found -- nothing to write.", file=sys.stderr)
        sys.exit(1)

    env_lines: list[str] = []
    for slot, info in slots.items():
        upper = slot.upper().replace("-", "_")
        env_lines += [
            f"# {slot}: pod {info['pod_id']} ({info.get('gpu_type') or '?'}, {info['status']})",
            f"{upper}_AGENT_MODEL_PROVIDER=runpod-vllm",
            f"{upper}_AGENT_MODEL_BASE_URL={info['base_url']}",
            f"{upper}_AGENT_MODEL_NAME={info['hf_model']}",
            f"{upper}_AGENT_MODEL_API_KEY_ENVS=VLLM_API_KEY",
        ]

    block = "\n".join(
        [BLOCK_BEGIN,
         f"# synced {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -- re-run the script to refresh; do not edit by hand"]
        + env_lines
        + ["# Protective flags (see ../docs/benchmark_guides.md, section 4.4):"]
        + PROTECTIVE_FLAGS
        + [BLOCK_END]
    ) + "\n"

    if args.dry_run:
        print(f"\n--dry-run: would write this managed block to {ENV_PATH}:\n")
        print(block, end="")
        return

    existing = ENV_PATH.read_text(encoding="utf-8") if ENV_PATH.exists() else ""

    # Warn about QWEN* base-url exports living OUTSIDE the managed block --
    # the appended block wins on `source` order, but duplicates confuse.
    outside = existing
    if BLOCK_BEGIN in outside and BLOCK_END in outside:
        pre, _, rest = outside.partition(BLOCK_BEGIN)
        _, _, post = rest.partition(BLOCK_END)
        outside = pre + post
    for line in outside.splitlines():
        if re.match(r"\s*(export\s+)?QWEN\w*_AGENT_MODEL_BASE_URL=", line):
            print(f"WARNING: .env sets a QWEN base URL outside the managed block "
                  f"({line.split('=')[0].strip()}) -- the managed block at the end "
                  f"overrides it on source; consider removing the manual line.",
                  file=sys.stderr)

    if BLOCK_BEGIN in existing and BLOCK_END in existing:
        pre, _, rest = existing.partition(BLOCK_BEGIN)
        _, _, post = rest.partition(BLOCK_END)
        post = post.lstrip("\n")
        updated = pre.rstrip("\n") + "\n\n" + block + (post if post else "")
        action = "replaced managed block in"
    else:
        updated = (existing.rstrip("\n") + "\n\n" if existing else "") + block
        action = "appended managed block to"
    ENV_PATH.write_text(updated, encoding="utf-8")

    print(f"\n{action} {ENV_PATH} ({len(slots)} slot(s))")
    print("\nsource .env")
    print("./scripts/alignmentgraph-isd-bench/5_run_ladder.sh")


if __name__ == "__main__":
    main()
