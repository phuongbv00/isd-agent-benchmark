#!/usr/bin/env python3
"""Post-hoc offline constructive-alignment scoring over a benchmark run dir.

Walks a results run directory (flat single-run or nested dataset layout),
scores every ``<agent_id>_output.json`` with the deterministic, threshold-free
:class:`isd_evaluator.metrics.alignment.AlignmentEvaluator` (primary endpoint:
``objective_assessment_alignment`` — continuous mean-max cosine), writes an
``alignment_scores.json`` next to the outputs of each scenario, and prints a
per-agent summary table (also written as CSV).

The primary encoder is an OpenAI-compatible ``/v1/embeddings`` endpoint
(``--encoder api``, e.g. a RunPod pod running
``vllm serve nvidia/llama-embed-nemotron-8b --task embed``) with a persistent
vector cache under the results dir — re-runs and sensitivity passes never
re-embed a text. ``--encoder tfidf`` (pure-Python, no network) is the
encoder-sensitivity fallback; Bloom levels always come from the rule-based
verb lexicon (no LLM anywhere in the measurement).

Examples:
  EMBED_BASE_URL=https://<pod>-8000.proxy.runpod.net/v1 \
      python scripts/alignmentgraph-isd-bench/06_score_alignment.py results/test_90_...
  python scripts/alignmentgraph-isd-bench/06_score_alignment.py results/test_90_... \
      --encoder tfidf --suffix tfidf

Re-run the same command after an interruption: complete per-scenario artifacts
are skipped, while corrupt or incomplete artifacts are repaired. Use
``--overwrite`` only for an intentional full re-score.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "evaluator" / "src"))

from isd_evaluator.metrics.alignment import (  # noqa: E402
    AlignmentEvaluator,
    OpenAIAPIEncoder,
    SentenceTransformerEncoder,
    TfidfCharNgramEncoder,
    reaggregate,
)

DEFAULT_EMBED_MODEL = "nvidia/llama-embed-nemotron-8b"

#: The sensitivity design, as (encoder kind, model, artifact suffix). Selecting
#: a preset ties the model to its suffix so an artifact can never end up
#: labelled with an encoder that did not produce it.
ENCODER_PRESETS = {
    # no suffix: the primary artifact, the one pooling reads
    "nemotron": ("api", "nvidia/llama-embed-nemotron-8b", ""),
    "bgem3":    ("api", "BAAI/bge-m3", "bgem3"),
    "granite":  ("api", "ibm-granite/granite-embedding-311m-multilingual-r2", "granite"),
    # lexical floor: pure CPU, no pod, bit-wise deterministic
    "tfidf":    ("tfidf", "tfidf-char-ngram", "tfidf"),
}

#: The encoder that owns the empty suffix — i.e. writes ``alignment_scores.json``,
#: the only file 07_pool_ladder_runs.py reads into ``pooled["alignment"]``.
#: 05_sync_embed_pod_env.py imports this to decide which slot the legacy
#: EMBED_BASE_URLS/EMBED_MODEL aliases point at.
PRIMARY_ENCODER = "nemotron"

#: suffix -> preset key, for discovering which encoders a results dir already
#: has on disk (07's --auto-encoders). The primary's empty suffix is excluded:
#: an unsuffixed file is the primary by definition, not a sweep arm.
SUFFIX_TO_KEY = {
    suffix: key for key, (_kind, _model, suffix) in ENCODER_PRESETS.items() if suffix
}


def encoder_slot(key: str) -> str:
    """Preset key -> env-var slot prefix, e.g. 'bgem3' -> 'BGEM3'.

    Same rule as the ladder's ``slot_upper`` (scripts/4_run_benchmark.sh), so
    the embed env-quad reads exactly like the agent one:
    ``<SLOT>_EMBED_BASE_URLS`` next to ``<SLOT>_AGENT_MODEL_BASE_URLS``.
    """
    return key.upper().replace("-", "_")


@dataclass
class EncoderSpec:
    """One arm of the encoder sweep, fully resolved before any scoring starts.

    ``available=False`` means this arm is skipped for the whole invocation (no
    endpoint, or the endpoint is down) — availability is an encoder-level
    property, so an arm either covers every run dir or none of them. Half-
    covered arms are the dangerous case: pooling would read the partial
    coverage as real data.
    """

    key: str
    kind: str                 # "api" | "st" | "tfidf"
    model: str
    suffix: str               # "" for the primary
    label: str                # what lands in the score file's "encoder" field
    revision: str | None = None   # weights revision of THIS arm's model
    base_urls: list[str] = field(default_factory=list)
    api_key_env: str = "VLLM_API_KEY"
    available: bool = True
    skip_reason: str | None = None

    @property
    def scores_name(self) -> str:
        return f"alignment_scores.{self.suffix}.json" if self.suffix else "alignment_scores.json"

    @property
    def summary_name(self) -> str:
        return f"alignment_summary.{self.suffix}.csv" if self.suffix else "alignment_summary.csv"


def served_model(base_url: str, api_key: str | None) -> str | None:
    """Ask the endpoint which model it serves; None if unreachable.

    Worth a round-trip before scoring: the embedding cache is keyed by
    sha1(model + text), so scoring against a pod that serves something other
    than --embed-model writes the wrong vectors under the right-looking key and
    mislabels the artifact. Both failures are silent.
    """
    import urllib.error  # noqa: PLC0415 - only needed on this path
    import urllib.request  # noqa: PLC0415

    req = urllib.request.Request(
        base_url.rstrip("/") + "/models",
        headers={"Authorization": f"Bearer {api_key or 'EMPTY'}",
                 # the RunPod proxy 403s the default urllib UA
                 "User-Agent": "curl/8.4.0 (isd-agent-benchmark)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        ids = [m.get("id") for m in data.get("data", []) if m.get("id")]
        return ids[0] if ids else None
    except Exception:  # noqa: BLE001 - unreachable is a warning, not a crash
        return None

def parse_ngram_range(text: str) -> tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", text or "")
    if not match:
        raise SystemExit(f"--tfidf-ngram-range expects LO-HI, got {text!r}")
    lo, hi = int(match.group(1)), int(match.group(2))
    if lo < 1 or hi < lo:
        raise SystemExit(f"--tfidf-ngram-range needs 1 <= LO <= HI, got {lo}-{hi}")
    return lo, hi


def _tfidf_label(model: str, ngram: tuple[int, int]) -> str:
    return model if ngram == (2, 3) else f"{model}({ngram[0]},{ngram[1]})"


def resolve_encoder_specs(args) -> list[EncoderSpec]:
    """Selected encoders, each with its endpoints resolved. No network yet.

    Selection: --all-encoders / --encoder-presets / --encoder-preset. With none
    of them this is the single-encoder legacy path, driven by --encoder /
    --embed-model / --suffix exactly as before.

    Endpoint precedence per arm (slot ``S = encoder_slot(key)``):
      1. --embed-base-urls-for S=<csv>
      2. <S>_EMBED_BASE_URLS          (written by 05_sync_embed_pod_env.py)
      3. the legacy --embed-base-url / EMBED_BASE_URLS / EMBED_BASE_URL, but
         only when exactly one api arm is selected — with several arms a single
         shared URL cannot be the right endpoint for all of them, and silently
         scoring bge-m3 against the Nemotron pod is exactly the cache-poisoning
         the served_model() guard exists to prevent.
      4. --embed-cache-only: a syntactically valid URL that is never called.
    """
    ngram = parse_ngram_range(args.tfidf_ngram_range)
    overrides: dict[str, str] = {}
    for spec in args.embed_base_urls_for:
        if "=" not in spec:
            raise SystemExit(f"--embed-base-urls-for expects SLOT=URL1,URL2 got: {spec}")
        slot, urls = spec.split("=", 1)
        overrides[slot.strip().upper().replace("-", "_")] = urls
    rev_overrides: dict[str, str] = {}
    for spec in args.embed_revision_for:
        if "=" not in spec:
            raise SystemExit(f"--embed-revision-for expects SLOT=REVISION got: {spec}")
        slot, rev = spec.split("=", 1)
        rev_overrides[slot.strip().upper().replace("-", "_")] = rev.strip()

    keys: list[str] = []
    if args.all_encoders:
        keys += list(ENCODER_PRESETS)
    for raw in (args.encoder_presets or "").split(","):
        if raw.strip():
            keys.append(raw.strip())
    if args.encoder_preset:
        keys.append(args.encoder_preset)
    seen: set[str] = set()
    keys = [k for k in keys if not (k in seen or seen.add(k))]
    unknown = [k for k in keys if k not in ENCODER_PRESETS]
    if unknown:
        raise SystemExit(
            f"unknown encoder preset(s) {unknown}; choose from {sorted(ENCODER_PRESETS)}"
        )

    # -- legacy single-encoder path -----------------------------------------
    if not keys:
        if args.encoder == "tfidf":
            label = _tfidf_label("tfidf-char-ngram", ngram)
        elif args.encoder == "st":
            label = f"{args.embed_model} (local)"
        else:
            label = args.embed_model
        raw_urls = args.embed_base_url or ""
        return [EncoderSpec(
            key=SUFFIX_TO_KEY.get(args.suffix or "", args.suffix or PRIMARY_ENCODER),
            kind=args.encoder,
            model="tfidf-char-ngram" if args.encoder == "tfidf" else args.embed_model,
            suffix=args.suffix or "",
            label=label,
            # tfidf is pure code in this repo — an HF weights revision is
            # meaningless for it and would be a lie in the artifact.
            revision=None if args.encoder == "tfidf" else args.embed_model_revision,
            base_urls=[u.strip() for u in raw_urls.split(",") if u.strip()],
            api_key_env=args.embed_api_key_env,
        )]

    # -- sweep path ----------------------------------------------------------
    n_api = sum(1 for k in keys if ENCODER_PRESETS[k][0] == "api")
    specs: list[EncoderSpec] = []
    for key in keys:
        kind, model, suffix = ENCODER_PRESETS[key]
        slot = encoder_slot(key)
        label = _tfidf_label(model, ngram) if kind == "tfidf" else model
        # Revision is PER ARM. A sweep scores several different models in one
        # invocation, so the single --embed-model-revision can only ever be
        # right for one of them; stamping it on all four would mislabel three
        # artifacts. Precedence mirrors the endpoint one, and the legacy flag
        # is honoured only when it is unambiguous (exactly one api arm).
        revision = (
            rev_overrides.get(slot)
            or os.environ.get(f"{slot}_EMBED_REVISION")
            or (args.embed_model_revision if kind == "api" and n_api == 1 else None)
        ) if kind != "tfidf" else None
        spec = EncoderSpec(key=key, kind=kind, model=model, suffix=suffix,
                           label=label, revision=revision)

        env_model = os.environ.get(f"{slot}_EMBED_MODEL")
        if env_model and env_model != model:
            raise SystemExit(
                f"{slot}_EMBED_MODEL is {env_model!r} but preset {key!r} is {model!r}. "
                "The .env block is stale — re-run "
                "scripts/alignmentgraph-isd-bench/05_sync_embed_pod_env.py."
            )
        spec.api_key_env = (
            os.environ.get(f"{slot}_EMBED_API_KEY_ENV")
            or args.embed_api_key_env
            or "VLLM_API_KEY"
        )
        if kind != "api":
            specs.append(spec)          # tfidf / st need no endpoint
            continue

        raw = overrides.get(slot) or os.environ.get(f"{slot}_EMBED_BASE_URLS")
        if not raw and n_api == 1:
            raw = args.embed_base_url
        if not raw and args.embed_cache_only:
            raw = "http://embed-cache-only.invalid/v1"
        if raw:
            spec.base_urls = [u.strip() for u in raw.split(",") if u.strip()]
        if not spec.base_urls:
            spec.available = False
            spec.skip_reason = (
                f"no endpoint (run 05_sync_embed_pod_env.py to set "
                f"{slot}_EMBED_BASE_URLS, or pass --embed-cache-only)"
            )
        specs.append(spec)

    if args.embed_model_revision and n_api > 1:
        raise SystemExit(
            "--embed-model-revision (or EMBED_MODEL_REVISION) is a single "
            f"revision but {n_api} api arms are selected — one revision string "
            "cannot pin the weights of several different models, and stamping it "
            "on all of them mislabels every artifact but one. Use "
            "--embed-revision-for SLOT=REVISION (or <SLOT>_EMBED_REVISION) per arm: "
            + ", ".join(f"{encoder_slot(s.key)}=..." for s in specs if s.kind == "api")
        )
    return specs


def probe_specs(specs: list[EncoderSpec], api_key_of, cache_only: bool,
                legacy: bool) -> None:
    """Confirm each api arm serves what we are about to label it with.

    Done ONCE per invocation, not per run dir. A served-model mismatch is a
    hard error: the cache is keyed by sha1(model[@revision] + text), so scoring
    against the wrong pod writes wrong vectors under a right-looking key AND
    mislabels the score files — both silent.

    EVERY endpoint of the arm is probed, not just the first. ``encode()``
    load-balances batches round-robin across all of them, so a second pod
    serving a different model poisons the cache exactly as thoroughly as the
    first one would — checking only ``base_urls[0]`` would leave every pod
    after it unguarded.

    An unreachable endpoint is treated differently on the two paths. On the
    sweep path it marks the arm unavailable and the sweep goes on without it
    (the point of a sweep is that arms are independent). On the legacy
    single-encoder path it stays a warning and scoring proceeds, exactly as
    before — that path has no other arm to fall back to.
    """
    for spec in specs:
        if spec.kind != "api" or not spec.available or cache_only or not spec.base_urls:
            continue
        for url in spec.base_urls:
            live = served_model(url, api_key_of(spec))
            if live is None:
                if legacy:
                    print(f"WARNING: could not read {url}/models — pod up? "
                          f"{spec.api_key_env} set?", file=sys.stderr)
                    continue
                spec.available = False
                spec.skip_reason = (
                    f"endpoint unreachable at {url} "
                    f"(pod booting? {spec.api_key_env} unset?)"
                )
                break
            if live != spec.model:
                raise SystemExit(
                    f"[{spec.key}] endpoint {url} serves {live!r} but the "
                    f"encoder is {spec.model!r}. Re-sync the pod env "
                    f"(05_sync_embed_pod_env.py) or fix "
                    f"{encoder_slot(spec.key)}_EMBED_BASE_URLS."
                )


COMPONENTS = [
    "objective_assessment_alignment",
    "objective_activity_alignment",
    "objective_evaluation_alignment",
    "objective_cognitive_congruence",
    "porter_mean",
    "webb_bloom_consistency",
    "assessment_precision",
]


def find_scenario_dirs(run_dir: Path) -> list[Path]:
    """Directories that contain agent outputs (flat run dir or nested)."""
    if list(run_dir.glob("*_output.json")):
        return [run_dir]
    dirs = sorted(
        {path.parent for path in run_dir.glob("*/*/*_output.json")}
        | {path.parent for path in run_dir.glob("*/*_output.json")}
    )
    return dirs


def load_scenario(scenario_dir: Path, run_dir: Path, scenario_root: Path) -> dict | None:
    """Scenario dict for a scenario dir.

    Prefers the copy embedded in ``comparison_report.json`` (always matches
    the run), then falls back to ``<scenario_root>/<variant>/<dir_name>.json``.
    """
    report_path = scenario_dir / "comparison_report.json"
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            scenario = report.get("scenario")
            if isinstance(scenario, dict) and scenario:
                return scenario
        except (json.JSONDecodeError, OSError):
            pass
    variant = scenario_dir.parent.name if scenario_dir != run_dir else None
    candidates = [scenario_root / f"{scenario_dir.name}.json"]
    if variant:
        candidates.insert(0, scenario_root / variant / f"{scenario_dir.name}.json")
    candidates += sorted(scenario_root.glob(f"*/{scenario_dir.name}.json"))
    for candidate in candidates:
        if candidate.exists():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
    return None


def score_scenario_dir(
    scenario_dir: Path,
    evaluator: AlignmentEvaluator,
    scenario: dict | None,
    agents_filter: set[str] | None,
) -> tuple[dict[str, dict], dict[str, str]]:
    """(scores, missing_outputs) for one scenario dir.

    ``missing_outputs`` is the other half of the roster and is what makes an
    RQ2 failure policy possible at all. ``run_benchmark.py`` writes
    ``<agent>_output.json`` only on success but ``<agent>_log.txt`` on BOTH
    paths, so a log without an output is an agent that was attempted and
    crashed. Without recording it, a crashed agent simply vanishes from the
    scenario and pooling's paired intersection silently drops the scenario —
    i.e. capability failures would improve an agent's alignment mean instead
    of costing it anything.
    """
    scores: dict[str, dict] = {}
    missing: dict[str, str] = {}
    for output_path in sorted(scenario_dir.glob("*_output.json")):
        agent_id = output_path.stem[: -len("_output")]
        if agents_filter and agent_id not in agents_filter:
            continue
        try:
            addie_output = json.loads(output_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"    ! {agent_id}: unreadable output ({exc})", file=sys.stderr)
            missing[agent_id] = f"unreadable output: {exc}"
            continue
        score = evaluator.evaluate(addie_output, scenario)
        scores[agent_id] = score.to_dict()
    for log_path in sorted(scenario_dir.glob("*_log.txt")):
        agent_id = log_path.stem[: -len("_log")]
        if agents_filter and agent_id not in agents_filter:
            continue
        if agent_id in scores or agent_id in missing:
            continue
        missing[agent_id] = "attempted, no *_output.json (agent run failed)"
    if missing:
        print(f"    ! no score for {sorted(missing)} (recorded as missing_outputs)",
              file=sys.stderr)
    return scores, missing


def expected_agent_ids(
    scenario_dir: Path,
    agents_filter: set[str] | None,
) -> set[str]:
    """Agents attempted in this scenario, after applying ``--agents``.

    Successful attempts have an output; failed attempts still have a log.
    Their union is the roster a complete alignment artifact must account for
    through either ``agents`` or ``missing_outputs``.
    """
    ids = {
        path.stem[: -len("_output")]
        for path in scenario_dir.glob("*_output.json")
    }
    ids.update(
        path.stem[: -len("_log")]
        for path in scenario_dir.glob("*_log.txt")
    )
    return ids if agents_filter is None else ids & agents_filter


def reusable_scores_payload(
    scores_path: Path,
    spec: EncoderSpec,
    expected_agents: set[str],
) -> tuple[dict | None, str | None]:
    """Load a complete compatible artifact, or explain why it needs re-score.

    A mere ``Path.exists()`` is not a safe resume marker: a killed process can
    leave truncated JSON, and a previous ``--agents`` subset can leave a valid
    but incomplete artifact. Both cases are repaired automatically.
    """
    try:
        payload = json.loads(scores_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return None, f"unreadable cached artifact: {exc}"
    if not isinstance(payload, dict):
        return None, "cached artifact is not a JSON object"
    if payload.get("encoder") != spec.label:
        return None, (
            f"encoder mismatch: {payload.get('encoder')!r} != {spec.label!r}"
        )
    if payload.get("encoder_revision") != spec.revision:
        return None, (
            "encoder revision mismatch: "
            f"{payload.get('encoder_revision')!r} != {spec.revision!r}"
        )
    agents = payload.get("agents")
    missing = payload.get("missing_outputs")
    if not isinstance(agents, dict) or not isinstance(missing, dict):
        return None, "cached artifact lacks agents/missing_outputs objects"
    overlap = set(agents) & set(missing)
    if overlap:
        return None, f"agent(s) recorded as both scored and missing: {sorted(overlap)}"
    recorded_agents = set(agents) | set(missing)
    if recorded_agents != expected_agents:
        absent = sorted(expected_agents - recorded_agents)
        extra = sorted(recorded_agents - expected_agents)
        return None, f"agent roster mismatch: absent={absent}, extra={extra}"
    return payload, None


def write_json_atomic(path: Path, payload: dict) -> None:
    """Publish a complete JSON artifact with one atomic rename."""
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def summarize(per_agent: dict[str, list[dict]]) -> list[tuple[str, int, dict[str, float | None]]]:
    rows = []
    for agent_id in sorted(per_agent):
        entries = per_agent[agent_id]
        means: dict[str, float | None] = {}
        for component in COMPONENTS:
            values = [e[component] for e in entries if e.get(component) is not None]
            means[component] = statistics.fmean(values) if values else None
        sanity_values = [
            e["details"]["sanity"]["lexicon_vs_declared_agreement"]
            for e in entries
            if e.get("details", {}).get("sanity", {}).get("lexicon_vs_declared_agreement")
            is not None
        ]
        means["sanity_agreement"] = (
            statistics.fmean(sanity_values) if sanity_values else None
        )
        rows.append((agent_id, len(entries), means))
    rows.sort(key=lambda r: -(
        r[2]["objective_assessment_alignment"]
        if r[2]["objective_assessment_alignment"] is not None else -1
    ))
    return rows


#: (COMPONENTS key or extra, console column label) — one entry per column.
_TABLE_COLUMNS = [
    ("objective_assessment_alignment", "AsmAl"),
    ("objective_activity_alignment", "ActAl"),
    ("objective_evaluation_alignment", "EvlAl"),
    ("objective_cognitive_congruence", "CogCon"),
    ("porter_mean", "Porter"),
    ("webb_bloom_consistency", "BloomC"),
    ("assessment_precision", "Prec"),
    ("sanity_agreement", "Sanity"),
]


def print_table(rows: list[tuple[str, int, dict[str, float | None]]]) -> None:
    def fmt(value: float | None) -> str:
        return f"{value:.3f}" if value is not None else "  -  "

    header = f"{'agent':30s} {'n':>3s} " + " ".join(
        f"{label:>6s}" for _, label in _TABLE_COLUMNS
    )
    print(header)
    print("-" * len(header))
    for agent_id, n, means in rows:
        cells = " ".join(f"{fmt(means[key]):>6s}" for key, _ in _TABLE_COLUMNS)
        print(f"{agent_id:30s} {n:3d} {cells}")


def write_csv(path: Path, rows: list[tuple[str, int, dict[str, float | None]]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["agent_id", "n"] + COMPONENTS + ["sanity_agreement"])
        for agent_id, n, means in rows:
            writer.writerow(
                [agent_id, n]
                + [
                    f"{means[c]:.4f}" if means[c] is not None else ""
                    for c in COMPONENTS + ["sanity_agreement"]
                ]
            )


#: Agent whose headline number the sweep matrix shows. Purely cosmetic — the
#: CSV and the score files carry every agent.
SWEEP_FOCUS_AGENT = "alignmentgraph-isd"


def write_sweep_csv(path: Path, rows: list[dict]) -> None:
    """Long-format roster of the whole sweep: one row per (encoder, run dir, agent).

    Nothing here is recomputed — these are the same per-agent means the
    per-run tables already show, just stacked with the encoder as a column so
    the sweep is inspectable in one file.
    """
    cols = ["encoder", "model", "suffix", "run_dir", "agent_id", "n"]
    cols += COMPONENTS + ["sanity_agreement"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for row in rows:
            writer.writerow([
                row.get(c) if c in ("encoder", "model", "suffix", "run_dir",
                                    "agent_id", "n")
                else (f"{row[c]:.4f}" if row.get(c) is not None else "")
                for c in cols
            ])
    print(f"\nWrote {path} ({len(rows)} rows)")


def print_sweep_matrix(specs: list[EncoderSpec], scored: list[EncoderSpec],
                       rows: list[dict], n_run_dirs: int) -> None:
    """One line per selected encoder: what ran, what didn't, and why."""
    focus: dict[str, tuple[float | None, int]] = {}
    for spec in scored:
        vals = [r for r in rows
                if r["encoder"] == spec.key and r["agent_id"] == SWEEP_FOCUS_AGENT]
        means = [r["objective_assessment_alignment"] for r in vals
                 if r.get("objective_assessment_alignment") is not None]
        focus[spec.key] = (
            statistics.fmean(means) if means else None,
            sum(r["n"] for r in vals),
        )
    print(f"\n=== encoder sweep summary ({len(specs)} selected, {len(scored)} scored, "
          f"{len(specs) - len(scored)} skipped)")
    header = (f"{'encoder':10s} {'model':44s} {'suffix':10s} {'status':10s} "
              f"{'runs':>5s} {'scen':>6s} {SWEEP_FOCUS_AGENT} AsmAl")
    print(header)
    print("-" * len(header))
    for spec in specs:
        model = spec.model if len(spec.model) <= 44 else spec.model[:41] + "..."
        suffix = spec.suffix or "(primary)"
        if spec in scored:
            mean, n_scen = focus[spec.key]
            cell = f"{mean:.3f}" if mean is not None else "-"
            print(f"{spec.key:10s} {model:44s} {suffix:10s} {'scored':10s} "
                  f"{n_run_dirs:5d} {n_scen:6d} {cell}")
        else:
            print(f"{spec.key:10s} {model:44s} {suffix:10s} {'SKIPPED':10s} "
                  f"{'-':>5s} {'-':>6s} {spec.skip_reason}")
    if len(scored) < len(specs):
        print("Skipped arms can be filled in later: the embedding cache makes a "
              "re-run nearly free.")


def _reaggregate_run(run_dir: Path, args, spec: EncoderSpec) -> int:
    """Recompute objective-level endpoints in every alignment_scores*.json
    under ``run_dir`` from the stored per-objective signals — no scoring."""
    scores_name = spec.scores_name
    scenario_dirs = find_scenario_dirs(run_dir)
    if args.limit is not None:
        scenario_dirs = scenario_dirs[: args.limit]
    per_agent: dict[str, list[dict]] = defaultdict(list)
    touched = 0
    for scenario_dir in scenario_dirs:
        scores_path = scenario_dir / scores_name
        if not scores_path.exists():
            continue
        payload = json.loads(scores_path.read_text(encoding="utf-8"))
        for agent_id, score in payload.get("agents", {}).items():
            reaggregate(score)
            per_agent[agent_id].append(score)
        write_json_atomic(scores_path, payload)
        touched += 1
    if not touched:
        raise SystemExit(f"No {scores_name} found under {run_dir} to re-aggregate.")
    rows = summarize(per_agent)
    print(f"\nRe-aggregated {touched} scenario dir(s) (no re-encode):")
    print_table(rows)
    write_csv(run_dir / spec.summary_name, rows)
    print(f"\nWrote {run_dir / spec.summary_name}")
    return rows




def build_encoder(spec: EncoderSpec, run_dir: Path, args):
    """Instantiate the encoder for one arm.

    The embedding cache path is derived from the MODEL, not the run dir, and
    lives at ``results/.embed_cache_<model_slug>.sqlite`` — shared across every
    run dir and every sweep arm of that model, which is what makes re-runs and
    additional arms nearly free. Distinct models get distinct files, so
    concurrent arms never collide. Within one file the arm's ``revision`` is
    part of the row key, so pinning a new revision re-embeds instead of
    serving the previous weights' vectors under the new label.
    """
    if spec.kind == "api":
        if not spec.base_urls and not args.embed_cache_only:
            raise SystemExit(
                "--encoder api needs --embed-base-url (or EMBED_BASE_URLS/"
                "EMBED_BASE_URL), e.g. a RunPod pod running "
                f"'vllm serve {DEFAULT_EMBED_MODEL} --task embed'. Pass "
                "--embed-cache-only to run purely off the embedding cache."
            )
        # cache-only needs a syntactically valid client it will never call
        base_urls = spec.base_urls or ["http://embed-cache-only.invalid/v1"]
        model_slug = re.sub(r"[^A-Za-z0-9.-]+", "-", spec.model)
        encoder = OpenAIAPIEncoder(
            base_urls=base_urls,
            model=spec.model,
            api_key=os.environ.get(spec.api_key_env),
            cache_path=run_dir.parent / f".embed_cache_{model_slug}.sqlite",
            cache_only=args.embed_cache_only,
            revision=spec.revision,
        )
        if args.embed_cache_only:
            print("  (cache-only: no endpoint will be contacted)")
        elif len(base_urls) > 1:
            print(f"  (load-balancing {len(base_urls)} embed endpoints)")
        return encoder
    if spec.kind == "st":
        return SentenceTransformerEncoder(spec.model)
    return TfidfCharNgramEncoder(ngram_range=parse_ngram_range(args.tfidf_ngram_range))


def score_run(run_dir: Path, args, spec: EncoderSpec, base_encoder=None):
    """Score one run dir with one encoder arm; returns the summary rows.

    ``base_encoder`` is built once per arm by main() and reused across run
    dirs (one HTTP client pool, one SQLite handle).
    """
    if args.reaggregate:
        return _reaggregate_run(run_dir, args, spec)

    encoder = base_encoder if base_encoder is not None else build_encoder(spec, run_dir, args)
    encoder_label = spec.label
    agents_filter = (
        {a.strip() for a in args.agents.split(",") if a.strip()} if args.agents else None
    )

    scenario_dirs = find_scenario_dirs(run_dir)
    if args.limit is not None:
        scenario_dirs = scenario_dirs[: args.limit]
    if not scenario_dirs:
        raise SystemExit(f"No *_output.json found under {run_dir}")

    evaluator = AlignmentEvaluator(encoder=encoder)

    scores_name = spec.scores_name
    per_agent: dict[str, list[dict]] = defaultdict(list)
    for scenario_dir in scenario_dirs:
        scores_path = scenario_dir / scores_name
        expected_agents = expected_agent_ids(scenario_dir, agents_filter)
        payload = None
        if scores_path.exists() and not args.overwrite:
            payload, invalid_reason = reusable_scores_payload(
                scores_path,
                spec,
                expected_agents,
            )
            if payload is not None:
                print(
                    f"  = {scenario_dir.name} "
                    f"(cached, {len(payload.get('agents', {}))} agents)"
                )
            else:
                print(
                    f"  ! {scenario_dir.name}: {invalid_reason}; rescoring",
                    file=sys.stderr,
                )
        if payload is None:
            scenario = load_scenario(scenario_dir, run_dir, args.scenario_root.resolve())
            if scenario is None:
                print(f"  ! {scenario_dir.name}: scenario JSON not found, "
                      "topics fall back to a single bucket", file=sys.stderr)
            scores, missing = score_scenario_dir(
                scenario_dir, evaluator, scenario, agents_filter)
            payload = {
                "scenario_dir": scenario_dir.name,
                "scenario_id": (scenario or {}).get("scenario_id"),
                "encoder": encoder_label,
                "encoder_revision": spec.revision,
                "bloom_classifier": "lexicon",
                "agents": scores,
                "missing_outputs": missing,
            }
            write_json_atomic(scores_path, payload)
            print(f"  + {scenario_dir.name} ({len(scores)} agents)")
        for agent_id, score in payload.get("agents", {}).items():
            per_agent[agent_id].append(score)

    rows = summarize(per_agent)
    print(f"\nAlignment summary over {len(scenario_dirs)} scenario dir(s) "
          f"(encoder: {encoder_label}):")
    print_table(rows)
    csv_path = run_dir / spec.summary_name
    write_csv(csv_path, rows)
    print(f"\nWrote {csv_path}")
    print("Note: 'Sanity' is the validation-evidence lexicon-vs-declared Bloom "
          "agreement; it never enters scoring.")
    return rows




def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score constructive alignment (deterministic, LLM-free) over a run dir."
    )
    parser.add_argument("run_dirs", type=Path, nargs="+", metavar="RUN_DIR",
                        help="One or more run dirs (globs expand in the shell).")
    parser.add_argument(
        "--encoder-preset", choices=sorted(ENCODER_PRESETS),
        help="Score with one of the sensitivity-design encoders: sets "
             "--encoder/--embed-model/--suffix together so the artifact label "
             "always matches the encoder that wrote it. Overrides those flags. "
             "Alias for a one-element --encoder-presets.",
    )
    parser.add_argument(
        "--encoder-presets", default=None, metavar="K1,K2,...",
        help="Score the SAME run dirs with several encoders in one invocation "
             f"(the sensitivity sweep), e.g. '{','.join(ENCODER_PRESETS)}'. Each "
             "arm resolves its own endpoint from <SLOT>_EMBED_BASE_URLS (written "
             "by 05_sync_embed_pod_env.py); an arm with no live endpoint is "
             "SKIPPED with a reason rather than failing the sweep — fill it in "
             "later, the embedding cache makes that nearly free.",
    )
    parser.add_argument(
        "--all-encoders", action="store_true",
        help=f"Shorthand for --encoder-presets {','.join(ENCODER_PRESETS)}.",
    )
    parser.add_argument(
        "--embed-base-urls-for", action="append", default=[], metavar="SLOT=URL1,URL2",
        help="Per-arm endpoint override, bypassing .env (repeatable), e.g. "
             "--embed-base-urls-for BGEM3=https://<pod>-8000.proxy.runpod.net/v1",
    )
    parser.add_argument(
        "--require-all", action="store_true",
        help="Treat an unavailable encoder arm as a hard error instead of "
             "skipping it (use when the sweep must be complete or not at all).",
    )
    parser.add_argument(
        "--sweep-csv", default="results/alignment_encoder_sweep.csv", metavar="PATH",
        help="Long-format roster of the whole sweep (one row per encoder x run "
             "dir x agent). Pass '' to disable.",
    )
    parser.add_argument(
        "--tfidf-ngram-range", default="2-3", metavar="LO-HI",
        help="Character n-gram range for the TF-IDF encoder. Recorded in the "
             "artifact's encoder label, so a non-default range is a genuinely "
             "different, still-offline encoder — useful as a no-cost second arm "
             "when verifying the cross-encoder agreement statistics.",
    )
    parser.add_argument(
        "--scenario-root", type=Path, default=REPO_ROOT / "scenarios",
        help="Scenario directory root (default: <repo>/scenarios).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Score at most N scenario dirs.")
    parser.add_argument(
        "--agents", default=None,
        help="Comma-separated agent ids to score (default: every *_output.json).",
    )
    parser.add_argument(
        "--encoder", choices=["api", "st", "tfidf"], default="api",
        help="Text encoder: 'api' (OpenAI-compatible /v1/embeddings endpoint, "
             "default — the primary protocol encoder), 'st' (same model via "
             "local sentence-transformers) or 'tfidf' (pure-Python fallback, "
             "encoder-sensitivity pass).",
    )
    parser.add_argument(
        "--embed-model", default=os.environ.get("EMBED_MODEL", DEFAULT_EMBED_MODEL),
        help="Embedding model name (served model for 'api', HF name for 'st'; "
             "env EMBED_MODEL).",
    )
    parser.add_argument(
        "--embed-cache-only", action="store_true",
        help="Fail instead of calling the endpoint when a text is missing from "
             "the embedding cache. Lets a re-score run with no pod up, as long "
             "as every text was already embedded by an earlier pass.",
    )
    parser.add_argument(
        "--embed-model-revision", default=os.environ.get("EMBED_MODEL_REVISION"),
        help="Weights revision (HF commit hash / tag) of the embedding model: "
             "recorded as 'encoder_revision' AND folded into the embedding-cache "
             "key, so a revision bump re-embeds instead of reusing the old "
             "weights' vectors. The model name alone does not pin weights (env "
             "EMBED_MODEL_REVISION). Applies to the single-encoder path; in a "
             "multi-arm sweep it is accepted only when exactly one api arm is "
             "selected — otherwise use --embed-revision-for / "
             "<SLOT>_EMBED_REVISION, since one revision string cannot be correct "
             "for several different models.",
    )
    parser.add_argument(
        "--embed-revision-for", action="append", default=[], metavar="SLOT=REVISION",
        help="Per-arm weights revision (repeatable), e.g. "
             "--embed-revision-for BGEM3=5617a9f61b028005a4858fdac845db406aefb181. "
             "Falls back to <SLOT>_EMBED_REVISION in the env.",
    )
    parser.add_argument(
        "--embed-base-url",
        default=os.environ.get("EMBED_BASE_URLS") or os.environ.get("EMBED_BASE_URL"),
        help="OpenAI-compatible base URL(s) for --encoder api — one, or a "
             "comma-separated list of pods (load-balanced round-robin, "
             "concurrent). Env EMBED_BASE_URLS (plural, preferred) or "
             "EMBED_BASE_URL.",
    )
    parser.add_argument(
        "--embed-api-key-env", default=os.environ.get("EMBED_API_KEY_ENV", "VLLM_API_KEY"),
        help="NAME of the env var holding the endpoint's API key (indirection, "
             "same pattern as the ladder's AGENT_MODEL_API_KEY_ENVS; "
             "env EMBED_API_KEY_ENV).",
    )
    parser.add_argument(
        "--suffix", default=None,
        help="Write alignment_scores.<suffix>.json instead of "
             "alignment_scores.json (sensitivity passes; pooling only reads "
             "the unsuffixed file).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-score scenario dirs that already have the target scores file.",
    )
    parser.add_argument(
        "--reaggregate", action="store_true",
        help="Do NOT re-score: read the existing alignment_scores.json and "
             "recompute only the objective-level endpoints from the stored "
             "per-objective signals (details.objectives). No encoder, no pod, "
             "instant. Use after an objective-level metric-definition change "
             "(e.g. dropping a criterion); a signal/encoder change still needs "
             "a full re-score with --overwrite.",
    )
    args = parser.parse_args()

    legacy = not (args.all_encoders or args.encoder_presets or args.encoder_preset)
    specs = resolve_encoder_specs(args)

    run_dirs = [d.resolve() for d in args.run_dirs]
    missing = [d for d in run_dirs if not d.is_dir()]
    if missing:
        raise SystemExit("Not a directory: " + ", ".join(str(d) for d in missing))

    # Confirm every endpoint really serves what we are about to label its
    # artifacts with — once per invocation, not per run dir.
    if not args.reaggregate:
        probe_specs(specs, lambda s: os.environ.get(s.api_key_env),
                    args.embed_cache_only, legacy)
    if args.require_all:
        blocked = [s for s in specs if not s.available]
        if blocked:
            raise SystemExit("--require-all: " + "; ".join(
                f"[{s.key}] {s.skip_reason}" for s in blocked))

    if not any(s.available for s in specs):
        for spec in specs:
            print(f"  - {spec.key}: SKIPPED — {spec.skip_reason}", file=sys.stderr)
        raise SystemExit("No encoder arm is runnable.")

    # One confirmation for the whole invocation, naming the arm(s) that will
    # overwrite the primary artifact pooling reads.
    writes_primary = [s.key for s in specs if s.available and not s.suffix]
    if writes_primary and not args.reaggregate:
        print(f"NOTE: {', '.join(writes_primary)} writes an unsuffixed file — this "
              "OVERWRITES alignment_scores.json, the artifact pooling reads.")
        if sys.stdin.isatty() and input("Continue? [y/N] ").strip().lower() != "y":
            return 1

    # Encoder-outer / run-dir-inner. Scenario artifacts are published atomically
    # and validated before reuse. If a pod dies mid-arm, re-running the same
    # command skips complete scenarios, repairs corrupt/incomplete artifacts,
    # and continues from the first unfinished scenario.
    sweep_rows: list[dict] = []
    scored: list[EncoderSpec] = []
    for spec in specs:
        if not spec.available:
            print(f"\n=== encoder [{spec.key}] SKIPPED — {spec.skip_reason}")
            continue
        rev = f" @{spec.revision}" if spec.revision else " (revision unpinned)"
        print(f"\n=== encoder [{spec.key}] {spec.label}{rev} -> {spec.scores_name}")
        base_encoder = None if args.reaggregate else build_encoder(spec, run_dirs[0], args)
        for i, run_dir in enumerate(run_dirs, 1):
            if len(run_dirs) > 1:
                print(f"\n--- [{i}/{len(run_dirs)}] {run_dir.name}")
            for agent_id, n, means in score_run(run_dir, args, spec, base_encoder):
                sweep_rows.append({"encoder": spec.key, "model": spec.model,
                                   "suffix": spec.suffix, "run_dir": run_dir.name,
                                   "agent_id": agent_id, "n": n, **means})
        scored.append(spec)

    if args.sweep_csv and sweep_rows:
        write_sweep_csv(Path(args.sweep_csv), sweep_rows)
    print_sweep_matrix(specs, scored, sweep_rows, len(run_dirs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
