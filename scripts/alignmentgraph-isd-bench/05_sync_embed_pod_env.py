"""Sync the RQ2 embedding pod's env vars from RunPod straight into .env.

Companion of scripts/3_sync_runpod_pods_env.py (the ladder version), for the
alignment-scoring embedding pod. The pod is created MANUALLY in the RunPod
console from the template ``isd-agent-bench-embed-nemotron-8b``
(https://console.runpod.io/deploy?template=3i3er665bh — vLLM
``--task embed --trust-remote-code`` serving ``nvidia/llama-embed-nemotron-8b``;
see ../docs/benchmark_guides.md, section 8). This script does the remaining
mechanical part: find EVERY live embed pod, confirm each served model against
the pod's own /v1/models, and REWRITE the managed block at the end of the
repo's .env file (between the >>>/<<< markers below) — so ``source .env``
gives 06_score_alignment.py its endpoint(s). Re-running replaces the block,
never duplicates it; nothing outside the markers (including the ladder's own
managed block) is touched.

Encoder is a swept dimension, so pods are grouped into one SLOT PER SERVED
MODEL — the same env-quad shape the ladder uses for agent models
(``<SLOT>_AGENT_MODEL_BASE_URLS`` -> ``<SLOT>_EMBED_BASE_URLS``). Several pods
in ONE slot still load-balance a single embedding space and are listed
comma-separated; pods serving DIFFERENT models are separate arms of the
sensitivity sweep and no longer an error. Slot names come from the scorer's
ENCODER_PRESETS, so the model -> slot mapping is defined in exactly one place.

It only ever READS from the RunPod API (GET /pods). Creating, stopping and
deleting pods stays a console operation.

Requires: RUNPOD_API_KEY in .env -- loaded automatically via python-dotenv.

Usage:
  python scripts/alignmentgraph-isd-bench/05_sync_embed_pod_env.py            # sync pods -> .env block
  python scripts/alignmentgraph-isd-bench/05_sync_embed_pod_env.py --dry-run  # print the block, write nothing
  source .env
  python scripts/alignmentgraph-isd-bench/06_score_alignment.py --encoder-presets nemotron,tfidf \
      --overwrite results/<run dir>

The managed block, with a Nemotron pod (two endpoints) and a bge-m3 pod up:

  EMBED_SLOTS=nemotron,bgem3,tfidf
  NEMOTRON_EMBED_BASE_URLS=https://<pod_a>-8000.proxy.runpod.net/v1,https://<pod_b>-8000.proxy.runpod.net/v1
  NEMOTRON_EMBED_MODEL=nvidia/llama-embed-nemotron-8b
  NEMOTRON_EMBED_API_KEY_ENV=VLLM_API_KEY
  BGEM3_EMBED_BASE_URLS=https://<pod_c>-8000.proxy.runpod.net/v1
  BGEM3_EMBED_MODEL=BAAI/bge-m3
  BGEM3_EMBED_API_KEY_ENV=VLLM_API_KEY

Endpoints are per slot only — there is no shared EMBED_BASE_URLS alias, because
one URL cannot be the right endpoint for several encoder arms.
<SLOT>_EMBED_API_KEY_ENV holds the NAME of the env var carrying the Bearer token
(indirection, same pattern as the ladder's AGENT_MODEL_API_KEY_ENVS).
VLLM_API_KEY itself must already be set in .env -- it is the token the
template's vLLM server was started with (same RunPod secret as the ladder),
not the RunPod API key.
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
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
BLOCK_BEGIN = ("# >>> embed pod (managed by "
               "scripts/alignmentgraph-isd-bench/05_sync_embed_pod_env.py) >>>")
BLOCK_END = "# <<< embed pod <<<"

try:
    from dotenv import load_dotenv
    load_dotenv(ENV_PATH)
except ImportError:
    pass  # Skip if dotenv is not available

#: Matched case-insensitively against the pod's vLLM start command (the model
#: is its first positional token) and the pod name.
EMBED_PATTERN = re.compile(r"nemotron|llama-embed|bge-m3|granite-embedding|--task embed", re.I)


def _load_scorer():
    """Import 06_score_alignment.py for its encoder registry.

    Via the spec loader because the filename starts with a digit (same trick
    as ablation/08_pool_ablation_runs.py). The point is that the model -> slot
    mapping lives in exactly one place: the scorer's ENCODER_PRESETS. A second
    copy here would drift the moment an encoder is added.
    """
    import importlib.util  # noqa: PLC0415 - only needed on this path

    path = Path(__file__).resolve().parent / "06_score_alignment.py"
    spec = importlib.util.spec_from_file_location("_scorer", path)
    module = importlib.util.module_from_spec(spec)
    # Register before exec: @dataclass resolves annotations through
    # sys.modules[cls.__module__], which is None for an unregistered module.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_SCORER = _load_scorer()
ENCODER_PRESETS = _SCORER.ENCODER_PRESETS
encoder_slot = _SCORER.encoder_slot

#: served model id -> preset key ('nvidia/llama-embed-nemotron-8b' -> 'nemotron')
MODEL_TO_KEY = {model: key for key, (_kind, model, _suffix) in ENCODER_PRESETS.items()}

#: Encoders that need no pod at all; always listed in EMBED_SLOTS so the sweep
#: picks them up whatever the pod situation is.
OFFLINE_KEYS = [key for key, (kind, _m, _s) in ENCODER_PRESETS.items() if kind != "api"]


def _slot_for(model_name: str) -> tuple[str, bool]:
    """(slot, is_known) for a served model id."""
    key = MODEL_TO_KEY.get(model_name)
    if key:
        return encoder_slot(key), True
    slug = re.sub(r"[^A-Za-z0-9]+", "_", (model_name or "unknown").split("/")[-1])
    return slug.upper().strip("_"), False


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


def _pod_cmd(pod: dict) -> str:
    cmd = pod.get("dockerStartCmd") or pod.get("dockerArgs") or ""
    if isinstance(cmd, list):
        cmd = " ".join(str(c) for c in cmd)
    return str(cmd)


def _pod_model_hint(pod: dict) -> str:
    """First positional token of the start command (the served HF model id)."""
    for token in _pod_cmd(pod).split():
        if not token.startswith("-") and "/" in token:
            return token
    env = pod.get("env") or {}
    if isinstance(env, dict) and env.get("MODEL_NAME"):
        return str(env["MODEL_NAME"])
    return ""


def _served_model(base_url: str) -> str | None:
    """Ask the pod's vLLM server which model it actually serves (needs
    VLLM_API_KEY). Authoritative over anything parsed from the pod record.
    Returns None if unreachable (pod booting, key unset, ...)."""
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
                        help="Print the managed block without writing files")
    parser.add_argument("--only-model", default=None,
                        help="Sync only pods whose served model id contains this "
                             "substring (e.g. 'bge-m3'). Optional: pods serving "
                             "different models now land in separate slots, so "
                             "several encoders can be up at once. Use this to "
                             "refresh one slot without touching the others.")
    parser.add_argument("--allow-unverified", action="store_true",
                        help="Also write pods whose /v1/models could not be read "
                             "(model id then comes from the RunPod record, which "
                             "is a NAME, not evidence of what is being served). "
                             "Off by default: an unverified pod in a slot is "
                             "load-balanced like any other and can write another "
                             "model's vectors into this model's cache.")
    args = parser.parse_args()

    data = _rest("/pods")
    pods = data if isinstance(data, list) else data.get("pods", [])

    found: list[dict] = []
    print(f"{len(pods)} pod(s) on the account:")
    for pod in pods:
        pod_id = pod.get("id", "?")
        name = pod.get("name", "")
        status = pod.get("desiredStatus", "?")
        model_hint = _pod_model_hint(pod)
        is_embed = bool(EMBED_PATTERN.search(_pod_cmd(pod))
                        or EMBED_PATTERN.search(name))
        tag = "embed" if is_embed else "-"
        print(f"  [{tag:5s}] {pod_id}  {status:8s}  model={model_hint or '?'}  name={name}")
        if is_embed:
            found.append(pod)

    if not found:
        print("\nERROR: no embed pod found (deploy one in the console from "
              "template isd-agent-bench-embed-nemotron-8b, "
              "https://console.runpod.io/deploy?template=3i3er665bh).",
              file=sys.stderr)
        sys.exit(1)

    # Verify each embed pod and file it under the slot of the model it serves.
    # Pods within ONE slot must serve the same model -- the encoder
    # load-balances batches across them, so a mismatch there would silently
    # corrupt the run. Pods serving DIFFERENT models are no longer an error:
    # they are separate encoder arms of the sensitivity sweep and get their
    # own <SLOT>_EMBED_* quad.
    endpoints: list[dict] = []
    for pod in found:
        pod_id = pod.get("id")
        status = pod.get("desiredStatus", "?")
        base_url = f"https://{pod_id}-{args.port}.proxy.runpod.net/v1"
        model_hint = _pod_model_hint(pod)
        served = _served_model(base_url)
        model_name = served or model_hint
        gpu = (pod.get("machine") or {}).get("gpuTypeId") or pod.get("gpuTypeId")
        if status != "RUNNING":
            print(f"WARNING: embed pod {pod_id} is {status}, not RUNNING -- "
                  "skipping (start it or re-run when RUNNING)", file=sys.stderr)
            continue
        if served and model_hint and served != model_hint:
            print(f"NOTE: pod {pod_id} record says {model_hint!r} but server "
                  f"serves {served!r}; using the server's id", file=sys.stderr)
        if not served:
            # The RunPod record's model is a pod NAME, not evidence of what is
            # being served, and the same-model guard below can only compare the
            # strings it is given. Writing an unconfirmed pod into a slot means
            # the scorer load-balances batches onto it and, if it actually
            # serves something else, poisons the cache under a right-looking
            # key. Default to leaving it out; the pod can be added by re-running
            # this script once it answers /v1/models.
            if not args.allow_unverified:
                print(f"WARNING: could not confirm pod {pod_id} via {base_url}/models "
                      f"(booting? VLLM_API_KEY unset?) -- EXCLUDING it from .env "
                      "(re-run when it is up, or pass --allow-unverified)",
                      file=sys.stderr)
                continue
            print(f"WARNING: could not confirm pod {pod_id} via {base_url}/models "
                  f"(booting? VLLM_API_KEY unset?) -- including {model_name!r} "
                  "unverified because --allow-unverified was passed", file=sys.stderr)
        if args.only_model and args.only_model.lower() not in (model_name or "").lower():
            print(f"  (skipping pod {pod_id}: serves {model_name!r}, "
                  f"not matching --only-model {args.only_model!r})")
            continue
        slot, known = _slot_for(model_name)
        if not known:
            print(f"WARNING: pod {pod_id} serves {model_name!r}, which matches no "
                  f"ENCODER_PRESETS entry -- filing it under slot {slot}; scoring "
                  "it needs explicit 06_score_alignment.py flags.", file=sys.stderr)
        endpoints.append({"pod_id": pod_id, "status": status, "gpu": gpu,
                          "base_url": base_url, "model": model_name,
                          "verified": bool(served), "slot": slot, "known": known})

    if not endpoints:
        print("\nERROR: no RUNNING, model-verified embed pod to sync. If a pod is "
              "up but its /v1/models did not answer, wait for it to finish loading "
              "(or check VLLM_API_KEY) and re-run; --allow-unverified overrides at "
              "the cost of the served-model guarantee.", file=sys.stderr)
        sys.exit(1)

    # Group into slots, preserving the ENCODER_PRESETS order so the block reads
    # in sweep order (primary first) rather than in RunPod API order.
    slots: dict[str, list[dict]] = {}
    for endpoint in endpoints:
        slots.setdefault(endpoint["slot"], []).append(endpoint)
    known_order = [encoder_slot(k) for k in ENCODER_PRESETS]
    ordered = [s for s in known_order if s in slots]
    ordered += [s for s in slots if s not in known_order]

    for slot in ordered:
        models = {e["model"] for e in slots[slot] if e["model"]}
        if len(models) > 1:
            print(f"\nERROR: slot {slot} has pods serving DIFFERENT models "
                  f"{sorted(models)} -- pods in one slot load-balance a single "
                  "embedding space, so they must serve the same model. Stop the "
                  f"odd pod(s) or re-run with --only-model {sorted(models)[0]}.",
                  file=sys.stderr)
            sys.exit(1)

    # Revisions are the one thing in this block a human sets and the API cannot
    # tell us, so carry the previous values forward instead of wiping them on
    # every re-sync (which would silently un-pin the weights).
    prev_env = ENV_PATH.read_text(encoding="utf-8") if ENV_PATH.exists() else ""
    prev_revisions = dict(
        re.findall(r"^\s*(?:export\s+)?(\w+)_EMBED_REVISION=(\S+)\s*$",
                   prev_env, re.MULTILINE)
    )

    lines: list[str] = []
    for slot in ordered:
        pods_here = slots[slot]
        model = next((e["model"] for e in pods_here if e["model"]), "")
        pod_desc = ", ".join(
            f"pod {e['pod_id']} ({e['gpu'] or '?'}, {e['status']}, "
            f"{'verified live' if e['verified'] else 'UNVERIFIED'})"
            for e in pods_here
        )
        lines += [
            f"# {slot.lower()}: {pod_desc}",
            f"{slot}_EMBED_BASE_URLS=" + ",".join(e["base_url"] for e in pods_here),
            f"{slot}_EMBED_MODEL={model}",
            f"{slot}_EMBED_API_KEY_ENV=VLLM_API_KEY",
            # The served model id does not pin weights, and the revision is not
            # discoverable from /v1/models — it is hand-set once and then
            # preserved here across re-syncs.
            f"{slot}_EMBED_REVISION={prev_revisions[slot]}" if slot in prev_revisions
            else f"# {slot}_EMBED_REVISION=<HF commit hash>  # set to pin weights"
                 " (also enters the embedding-cache key)",
        ]
    for key in OFFLINE_KEYS:
        lines.append(f"# {key}: no pod needed (pure CPU) -- listed in EMBED_SLOTS anyway")

    slot_keys = [s.lower() for s in ordered] + OFFLINE_KEYS
    # A slot with no matching preset has no --encoder-presets key to offer.
    preset_keys = [k for k in slot_keys if k in ENCODER_PRESETS]
    block = "\n".join([
        BLOCK_BEGIN,
        f"# synced {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -- "
        f"{len(ordered)} pod slot(s), {len(endpoints)} endpoint(s); "
        "re-run the script to refresh; do not edit by hand",
        f"EMBED_SLOTS={','.join(slot_keys)}",
        *lines,
        BLOCK_END,
    ]) + "\n"

    if args.dry_run:
        print(f"\n--dry-run: would write this managed block to {ENV_PATH}:\n")
        print(block, end="")
        return

    existing = prev_env

    # Warn about EMBED_* exports living OUTSIDE the managed block -- the
    # appended block wins on `source` order, but duplicates confuse.
    outside = existing
    if BLOCK_BEGIN in outside and BLOCK_END in outside:
        pre, _, rest = outside.partition(BLOCK_BEGIN)
        _, _, post = rest.partition(BLOCK_END)
        outside = pre + post
    for line in outside.splitlines():
        if re.match(r"\s*(export\s+)?(\w+_)?EMBED_"
                    r"(BASE_URLS?|MODEL|MODEL_REVISION|REVISION|API_KEY_ENV|SLOTS)=", line):
            print(f"WARNING: .env sets {line.split('=')[0].strip()} outside the "
                  f"managed block -- the managed block at the end overrides it "
                  f"on source; consider removing the manual line.", file=sys.stderr)

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

    print(f"\n{action} {ENV_PATH} ({len(ordered)} pod slot(s))")
    print("\nsource .env")
    print("python scripts/alignmentgraph-isd-bench/06_score_alignment.py "
          f"--encoder-presets {','.join(preset_keys)} --overwrite results/<run dir>")


if __name__ == "__main__":
    main()
