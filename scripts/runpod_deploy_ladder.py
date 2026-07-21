#!/usr/bin/env python3
"""Deploy/manage RunPod Serverless vLLM endpoints for the RQ1 model ladder.

Self-hosts the exact Qwen checkpoints for the ladder (0.8B / 2B / 4B / 9B)
as RunPod Serverless vLLM endpoints, instead of guessing OpenRouter model IDs
(see the TODO(user) markers in 6_run_ladder.sh). Each endpoint exposes an
OpenAI-compatible API that plugs directly into the benchmark's existing
--agent-model-provider/--agent-model-base-url/--agent-model-name/
--agent-model-api-key-envs flags -- no changes needed to 6_run_ladder.sh
itself, only different env-quad values.

Cost/speed design (see chat discussion, RunPod docs as of 2026-07):
  - Serverless, not Pods: pay per second of actual inference, scale to zero
    when idle. Fits this repo's "never auto-run benchmark" policy -- runs
    are manual/bursty, not continuous, so idle-billed Pods would waste money.
  - One endpoint per model size, GPU tier sized to that size (don't pay for
    a 9B-sized GPU while serving 0.8B).
  - A single shared NetworkVolume across all endpoints caches Hugging Face
    weights at /runpod-volume/huggingface, so the 3 repeated ladder runs
    (and re-launches) don't re-download weights.
  - workersMin=0 (scale to zero), low idleTimeout, so cost is purely
    per-request; no baseline "active worker" cost, which only pays off at a
    steady request rate this workload doesn't have.

THIS SCRIPT ONLY TALKS TO THE RUNPOD API (creates/lists/deletes endpoints,
templates, and a network volume). It does not run the benchmark and does not
call any LLM. Creating a scale-to-zero endpoint costs nothing until invoked,
but review the GPU tiers below before deploying -- verify exact HF model IDs
and RunPod gpuTypeIds first (see the `gpu-types` subcommand).

Requires: RUNPOD_API_KEY env var (https://www.runpod.io/console/user/settings).

Usage:
  # 1. List available GPU types to pick exact gpuTypeIds (verify TODOs below):
  python scripts/runpod_deploy_ladder.py gpu-types

  # 2. Create the shared model-cache volume once:
  python scripts/runpod_deploy_ladder.py create-volume --name qwen-ladder-cache \
      --size 100 --datacenter US-KS-2

  # 3. Deploy all 4 ladder slots against that volume:
  python scripts/runpod_deploy_ladder.py deploy --volume-id <id-from-step-2>

  # Deploy a single slot (e.g. re-deploying after a HF id fix):
  python scripts/runpod_deploy_ladder.py deploy --volume-id <id> --slots qwen9b

  # 4. Check endpoint health / worker status:
  python scripts/runpod_deploy_ladder.py status --manifest results/runpod_ladder_<ts>/manifest.json

  # 5. When the ladder is fully done, tear down endpoints (keep the volume
  #    if you'll re-run later; the volume itself has its own idle storage cost):
  python scripts/runpod_deploy_ladder.py teardown --manifest results/runpod_ladder_<ts>/manifest.json

Deploy prints (and writes to the manifest) the exact env-quad exports to
source before ./scripts/6_run_ladder.sh, e.g.:

  export QWEN08B_AGENT_MODEL_PROVIDER=runpod-vllm
  export QWEN08B_AGENT_MODEL_BASE_URL=https://api.runpod.ai/v2/<endpoint_id>/openai/v1
  export QWEN08B_AGENT_MODEL_NAME=Qwen/Qwen3.5-0.8B
  export QWEN08B_AGENT_MODEL_API_KEY_ENVS=RUNPOD_API_KEY

(api_spec defaults to openai_compatible automatically once base_url is set;
 see shared/llm/config.py.)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime

REST_BASE = "https://rest.runpod.io/v1"
GRAPHQL_URL = "https://api.runpod.io/graphql"
VLLM_IMAGE = "runpod/worker-vllm:stable-cuda12.1.0"

# ---------------------------------------------------------------------------
# Ladder slot defaults. HF model IDs and GPU tiers are TODO(user): VERIFY --
# same caution as the OpenRouter IDs in 6_run_ladder.sh. Check the exact
# checkpoint names on https://huggingface.co/Qwen and confirm VRAM budget
# against `gpu-types` output before deploying.
# ---------------------------------------------------------------------------
DEFAULT_SLOTS = {
    "qwen08b": {
        "hf_model": "Qwen/Qwen3.5-0.8B",  # TODO(user): verify exact HF id
        "gpu_type_ids": ["NVIDIA GeForce RTX 4090"],  # TODO(user): verify id via gpu-types
        "container_disk_gb": 30,
    },
    "qwen2b": {
        "hf_model": "Qwen/Qwen3.5-2B",  # TODO(user): verify exact HF id
        "gpu_type_ids": ["NVIDIA GeForce RTX 4090"],  # TODO(user): verify id via gpu-types
        "container_disk_gb": 30,
    },
    "qwen4b": {
        "hf_model": "Qwen/Qwen3.5-4B",  # TODO(user): verify exact HF id
        "gpu_type_ids": ["NVIDIA GeForce RTX 4090"],  # TODO(user): verify id via gpu-types
        "container_disk_gb": 40,
    },
    "qwen9b": {
        "hf_model": "Qwen/Qwen3.5-9B",  # TODO(user): verify exact HF id
        # Bigger VRAM margin for 9B fp16 weights + vLLM KV cache under
        # concurrent benchmark load. TODO(user): verify id via gpu-types.
        "gpu_type_ids": ["NVIDIA RTX A6000"],
        "container_disk_gb": 60,
    },
}


def _api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        print("ERROR: RUNPOD_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)
    return key


def _rest(method: str, path: str, body: dict | None = None) -> dict:
    url = f"{REST_BASE}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {_api_key()}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        print(f"ERROR {method} {path}: HTTP {e.code}\n{detail}", file=sys.stderr)
        sys.exit(1)


def _graphql(query: str) -> dict:
    url = f"{GRAPHQL_URL}?api_key={_api_key()}"
    req = urllib.request.Request(
        url, data=json.dumps({"query": query}).encode("utf-8"), method="POST"
    )
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        print(f"ERROR graphql: HTTP {e.code}\n{detail}", file=sys.stderr)
        sys.exit(1)


def cmd_gpu_types(_args: argparse.Namespace) -> None:
    result = _graphql("query { gpuTypes { id displayName memoryInGb } }")
    gpu_types = result.get("data", {}).get("gpuTypes", [])
    for g in sorted(gpu_types, key=lambda g: g["memoryInGb"]):
        print(f"{g['memoryInGb']:>4} GB  {g['id']!r:45} ({g['displayName']})")


def cmd_create_volume(args: argparse.Namespace) -> None:
    body = {"name": args.name, "size": args.size, "dataCenterId": args.datacenter}
    result = _rest("POST", "/networkvolumes", body)
    print(f"Created network volume: id={result.get('id')} name={result.get('name')}")
    print(f"Pass this as --volume-id to `deploy`: {result.get('id')}")


def _slot_upper(slot: str) -> str:
    return slot.upper().replace("-", "_")


def _create_template(slot: str, cfg: dict, volume_id: str, args: argparse.Namespace) -> str:
    env = {
        "MODEL_NAME": cfg["hf_model"],
        "HF_HOME": "/runpod-volume/huggingface",
    }
    if args.hf_token:
        env["HUGGING_FACE_HUB_TOKEN"] = args.hf_token
    if args.max_model_len:
        env["MAX_MODEL_LEN"] = str(args.max_model_len)
    body = {
        "name": f"ladder-{slot}-{int(time.time())}",
        "imageName": VLLM_IMAGE,
        "isServerless": True,
        "containerDiskInGb": cfg["container_disk_gb"],
        "env": env,
    }
    result = _rest("POST", "/templates", body)
    template_id = result.get("id")
    if not template_id:
        print(f"ERROR: template creation for {slot} did not return an id: {result}", file=sys.stderr)
        sys.exit(1)
    return template_id


def _create_endpoint(slot: str, cfg: dict, template_id: str, volume_id: str, args: argparse.Namespace) -> str:
    body = {
        "name": f"ladder-{slot}",
        "templateId": template_id,
        "gpuTypeIds": cfg["gpu_type_ids"],
        "workersMin": 0,
        "workersMax": args.workers_max,
        "idleTimeout": args.idle_timeout,
        "networkVolumeId": volume_id,
    }
    result = _rest("POST", "/endpoints", body)
    endpoint_id = result.get("id")
    if not endpoint_id:
        print(f"ERROR: endpoint creation for {slot} did not return an id: {result}", file=sys.stderr)
        sys.exit(1)
    return endpoint_id


def cmd_deploy(args: argparse.Namespace) -> None:
    slots = args.slots.split(",") if args.slots else list(DEFAULT_SLOTS.keys())
    unknown = [s for s in slots if s not in DEFAULT_SLOTS]
    if unknown:
        print(f"ERROR: unknown slot(s) {unknown}; known: {list(DEFAULT_SLOTS)}", file=sys.stderr)
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("results", f"runpod_ladder_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    manifest = {"timestamp": timestamp, "volume_id": args.volume_id, "slots": {}}
    env_lines = []

    for slot in slots:
        cfg = DEFAULT_SLOTS[slot]
        print(f"[{slot}] creating template for {cfg['hf_model']} on {cfg['gpu_type_ids']} ...")
        template_id = _create_template(slot, cfg, args.volume_id, args)
        print(f"[{slot}] template_id={template_id}; creating endpoint ...")
        endpoint_id = _create_endpoint(slot, cfg, template_id, args.volume_id, args)
        base_url = f"https://api.runpod.ai/v2/{endpoint_id}/openai/v1"
        print(f"[{slot}] endpoint_id={endpoint_id}\n[{slot}] base_url={base_url}")

        manifest["slots"][slot] = {
            "hf_model": cfg["hf_model"],
            "gpu_type_ids": cfg["gpu_type_ids"],
            "template_id": template_id,
            "endpoint_id": endpoint_id,
            "base_url": base_url,
            "workers_max": args.workers_max,
            "idle_timeout": args.idle_timeout,
        }

        upper = _slot_upper(slot)
        env_lines += [
            f"export {upper}_AGENT_MODEL_PROVIDER=runpod-vllm",
            f"export {upper}_AGENT_MODEL_BASE_URL={base_url}",
            f"export {upper}_AGENT_MODEL_NAME={cfg['hf_model']}",
            f"export {upper}_AGENT_MODEL_API_KEY_ENVS=RUNPOD_API_KEY",
        ]

    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    env_path = os.path.join(out_dir, "ladder_env.sh")
    with open(env_path, "w", encoding="utf-8") as f:
        f.write("# Source this before ./scripts/6_run_ladder.sh\n")
        f.write("# Requires RUNPOD_API_KEY to already be exported (.env).\n")
        f.write("\n".join(env_lines) + "\n")

    print("\n==============================================================")
    print(f"Manifest: {manifest_path}")
    print(f"Env file: {env_path}")
    print("==============================================================")
    print("\nNext: verify the served model responds, then:")
    print(f"  source {env_path}")
    print("  ./scripts/6_run_ladder.sh")
    print("\nCold start note: the first request per endpoint downloads+loads")
    print("the model (cached on the network volume after that). Consider a")
    print("cheap warm-up request per endpoint before the timed benchmark run.")


def cmd_status(args: argparse.Namespace) -> None:
    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    for slot, info in manifest["slots"].items():
        endpoint_id = info["endpoint_id"]
        url = f"https://api.runpod.ai/v2/{endpoint_id}/health"
        req = urllib.request.Request(url)
        req.add_header("authorization", _api_key())
        try:
            with urllib.request.urlopen(req) as resp:
                health = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            print(f"[{slot}] HTTP {e.code}: {e.read().decode('utf-8', errors='replace')}")
            continue
        print(f"[{slot}] endpoint={endpoint_id} health={json.dumps(health)}")


def cmd_teardown(args: argparse.Namespace) -> None:
    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    for slot, info in manifest["slots"].items():
        endpoint_id = info["endpoint_id"]
        template_id = info["template_id"]
        print(f"[{slot}] deleting endpoint {endpoint_id} ...")
        _rest("DELETE", f"/endpoints/{endpoint_id}")
        print(f"[{slot}] deleting template {template_id} ...")
        _rest("DELETE", f"/templates/{template_id}")
    print("\nDone. The shared network volume was NOT deleted (reused across")
    print(f"re-runs); volume_id={manifest.get('volume_id')}. Delete it manually")
    print("via the RunPod console if you no longer need the cached weights.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("gpu-types", help="List RunPod GPU type ids/memory").set_defaults(func=cmd_gpu_types)

    p_vol = sub.add_parser("create-volume", help="Create the shared HF weight cache volume")
    p_vol.add_argument("--name", default="qwen-ladder-cache")
    p_vol.add_argument("--size", type=int, default=100, help="GB")
    p_vol.add_argument("--datacenter", default="US-KS-2")
    p_vol.set_defaults(func=cmd_create_volume)

    p_deploy = sub.add_parser("deploy", help="Create template+endpoint per ladder slot")
    p_deploy.add_argument("--volume-id", required=True)
    p_deploy.add_argument("--slots", default="", help="Comma-separated subset, e.g. qwen9b. Default: all")
    p_deploy.add_argument("--workers-max", type=int, default=3)
    p_deploy.add_argument("--idle-timeout", type=int, default=30, help="Seconds before scale-to-zero")
    p_deploy.add_argument("--hf-token", default=os.environ.get("HUGGING_FACE_HUB_TOKEN", ""))
    p_deploy.add_argument("--max-model-len", type=int, default=0)
    p_deploy.set_defaults(func=cmd_deploy)

    p_status = sub.add_parser("status", help="Check endpoint health from a deploy manifest")
    p_status.add_argument("--manifest", required=True)
    p_status.set_defaults(func=cmd_status)

    p_teardown = sub.add_parser("teardown", help="Delete endpoints+templates from a deploy manifest")
    p_teardown.add_argument("--manifest", required=True)
    p_teardown.set_defaults(func=cmd_teardown)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
