#!/usr/bin/env python3
"""Post-hoc offline constructive-alignment scoring over a benchmark run dir.

Walks a results run directory (flat single-run or nested dataset layout),
scores every ``<agent_id>_output.json`` with the deterministic, threshold-free
:class:`isd_evaluator.metrics.alignment.AlignmentEvaluator`, writes an
``alignment_scores.json`` next to the outputs of each scenario, and prints a
per-agent summary table (also written as CSV). The output is the flat panel of
``PANEL_SIGNALS`` — there is no primary endpoint and no composite.

**Two sensitivity axes, one per instrument family.** Encoders come from
:data:`ENCODER_PRESETS` (family A), Bloom classifiers from
:data:`BLOOM_PRESETS` (family B); each preset ties an instrument to its
artifact suffix, and to its BACKEND — so an artifact can never end up labelled
with an instrument that did not produce it.

**Only the 8B primary needs a GPU pod.** ``nemotron`` is served over an
OpenAI-compatible ``/v1/embeddings`` endpoint (e.g. a RunPod pod running
``vllm serve nvidia/llama-embed-nemotron-8b --task embed``), resolved from its
``<SLOT>_EMBED_*`` env quad (written by ``05_sync_embed_pod_env.py``). The
sweep arms run in-process: ``bgem3`` and ``granite`` through
sentence-transformers on the local GPU (measured on an M4 Pro: 126 and 226
texts/s, i.e. ~44 and ~25 minutes for the ladder's 335k unique texts),
``tfidf`` in pure Python. Every backend shares the same persistent vector cache
under the results dir, so re-runs and re-analysis never re-embed a text. No LLM
anywhere in the measurement, on any arm.

The sweep is a **star, not a grid**: every arm differs from the primary
configuration (``nemotron`` + ``bertbloom``) along exactly one axis. Asking for
a non-primary encoder *and* a non-primary Bloom arm in the same arm is an
error, not a silent cross-product — the question each axis answers ("does the
ranking depend on this instrument?") needs one deviation at a time.

Examples:
  # primary configuration only (the default when no preset flag is given)
  python scripts/alignmentgraph-isd-bench/06_score_alignment.py results/test_90_...
  # the full sensitivity sweep in one invocation (5 arms)
  python scripts/alignmentgraph-isd-bench/06_score_alignment.py results/test_90_... \
      --encoder-presets nemotron,bgem3,granite,tfidf --bloom-presets bertbloom,lexicon
  # everything except the primary encoder — no pod involved at all
  python scripts/alignmentgraph-isd-bench/06_score_alignment.py results/test_90_... \
      --encoder-presets bgem3,granite,tfidf --bloom-presets lexicon

Re-run the same command after an interruption: complete per-scenario artifacts
are skipped, while corrupt or incomplete artifacts are repaired. Use
``--overwrite`` only for an intentional full re-score.
"""

from __future__ import annotations

import argparse
import csv
import glob as globmod
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

# The endpoint this script needs (<SLOT>_EMBED_BASE_URLS) is WRITTEN into .env
# by 05_sync_embed_pod_env.py, so it has to be READ from there too. Without
# this, 05 would write the value, .env would visibly contain it, and this
# script would still report "no endpoint" and tell the user to re-run 05 —
# advice that cannot fix anything. Same guarded import as run_benchmark.py and
# 05 itself; load_dotenv does not override variables already exported in the
# shell, so an explicit export still wins.
try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass  # Skip if dotenv is not available

from isd_evaluator.metrics.alignment import (  # noqa: E402
    PANEL_SIGNALS,
    AlignmentEvaluator,
    LexiconBloomClassifier,
    OpenAIAPIEncoder,
    SentenceTransformerEncoder,
    TfidfCharNgramEncoder,
    TransformerBloomClassifier,
    reaggregate,
)

#: The sensitivity design, as (encoder kind, model, artifact suffix). Selecting
#: a preset ties the model to its suffix so an artifact can never end up
#: labelled with an encoder that did not produce it — and to its BACKEND, which
#: matters just as much: the embedding cache key is sha1(model[@revision]+text)
#: and does NOT record where a vector came from, so a model embedded both on a
#: pod and locally would mix two numerically different sources under one key,
#: invisibly. One backend per model, fixed here:
#:
#:   api    served over an OpenAI-compatible endpoint (needs a pod)
#:   local  sentence-transformers in-process (no pod; downloads once)
#:   tfidf  pure Python
#:
#: Only the 8B primary is served. The sweep arms are small enough to embed on
#: a laptop, which removes three quarters of the GPU-rental surface.
ENCODER_PRESETS = {
    # no suffix: the primary artifact, the one pooling reads
    "nemotron": ("api", "nvidia/llama-embed-nemotron-8b", ""),
    "bgem3":    ("local", "BAAI/bge-m3", "bgem3"),
    "granite":  ("local", "ibm-granite/granite-embedding-311m-multilingual-r2", "granite"),
    # lexical floor: pure CPU, no download, bit-wise deterministic
    "tfidf":    ("tfidf", "tfidf-char-ngram", "tfidf"),
}

#: The encoder that owns the empty suffix — i.e. writes ``alignment_scores.json``,
#: the only file 07_pool_ladder_runs.py reads into ``pooled["alignment"]``.
#: Also the default arm when no preset flag is given.
PRIMARY_ENCODER = "nemotron"

#: The Bloom axis, as (classifier kind, checkpoint path, artifact suffix) —
#: family B's counterpart of ENCODER_PRESETS. ``bertbloom`` is the primary arm:
#: a 6-way BERT fine-tuned on the EDM2022CLO corpus of Li et al. (2022), which
#: agrees with the released expert labels at kappa 0.926 on a held-out split
#: against the lexicon's 0.650. ``lexicon`` is the sweep arm — deterministic and
#: traceable to Anderson & Krathwohl, and derived from the taxonomy rather than
#: from labelled data, which is what makes it a real independence check for the
#: 3 Bloom-derived panel signals (the two arms agree at only kappa 0.665, so
#: neither is a mirror of the other).
#:
#: Consequence worth knowing before a run: the primary arm needs a checkpoint on
#: disk, so scoring is no longer zero-setup. Train it with
#: ``evaluator/scripts/train_bloom_6way_classifier.py``; without it every arm is
#: SKIPPED rather than silently falling back.
BLOOM_PRESETS = {
    "bertbloom": ("transformer", "models/bloom-bert-edm2022", ""),
    "lexicon":   ("lexicon", None, "lexicon"),
}

#: The Bloom arm that owns the empty suffix (pairs with PRIMARY_ENCODER to
#: form the primary configuration every other arm deviates from by one axis).
PRIMARY_BLOOM = "bertbloom"

#: suffix -> preset key, for discovering which arms a results dir already has
#: on disk (07's --auto-encoders / --auto-bloom). The primary's empty suffix is
#: excluded: an unsuffixed file is the primary by definition, not a sweep arm.
SUFFIX_TO_KEY = {
    suffix: key for key, (_kind, _model, suffix) in ENCODER_PRESETS.items() if suffix
}

BLOOM_SUFFIX_TO_KEY = {
    suffix: key for key, (_kind, _path, suffix) in BLOOM_PRESETS.items() if suffix
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
    """One arm of the sensitivity sweep, fully resolved before scoring starts.

    An arm is a pair (encoder, Bloom classifier). Under the star design at most
    one of the two deviates from the primary configuration, so at most one of
    ``encoder_suffix`` / ``bloom_suffix`` is non-empty and :attr:`suffix` is
    unambiguous.

    ``available=False`` means this arm is skipped for the whole invocation (no
    endpoint, the endpoint is down, or the Bloom checkpoint is not on disk) —
    availability is an arm-level property, so an arm either covers every run
    dir or none of them. Half-covered arms are the dangerous case: pooling
    would read the partial coverage as real data.
    """

    key: str
    kind: str                 # "api" | "local" | "tfidf"
    model: str
    encoder_suffix: str       # "" for the primary encoder
    label: str                # what lands in the score file's "encoder" field
    revision: str | None = None   # weights revision of THIS arm's model
    base_urls: list[str] = field(default_factory=list)
    api_key_env: str = "VLLM_API_KEY"
    bloom_key: str = PRIMARY_BLOOM
    bloom_kind: str = "lexicon"
    bloom_path: str | None = None
    bloom_suffix: str = ""    # "" for the primary Bloom arm
    available: bool = True
    skip_reason: str | None = None

    @property
    def suffix(self) -> str:
        """The artifact suffix of this arm — the axis it deviates on."""
        return self.encoder_suffix or self.bloom_suffix

    @property
    def arm_label(self) -> str:
        """Human-readable arm id for console output, e.g. 'nemotron+bertbloom'."""
        return f"{self.key}+{self.bloom_key}"

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
    than the arm's preset model writes the wrong vectors under the right-looking
    key and mislabels the artifact. Both failures are silent.
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


def resolve_arms(args) -> list[tuple[str, str]]:
    """The (encoder key, Bloom key) arms to score — the star, never the grid.

    Selection: --all-arms / --encoder-presets / --encoder-preset for family A,
    --bloom-presets / --bloom-preset for family B; with none of them the sweep
    is the single primary configuration (PRIMARY_ENCODER + PRIMARY_BLOOM).

    Every arm deviates from that configuration on **at most one axis**. A grid
    would double the cost and the surface to audit without answering anything
    the star does not: each axis asks "does the ranking depend on this
    instrument?", and that question needs one deviation at a time.
    """
    enc_keys: list[str] = []
    if args.all_arms or args.all_encoders:
        enc_keys += list(ENCODER_PRESETS)
    for raw in (args.encoder_presets or "").split(","):
        if raw.strip():
            enc_keys.append(raw.strip())
    if args.encoder_preset:
        enc_keys.append(args.encoder_preset)

    bloom_keys: list[str] = []
    if args.all_arms:
        bloom_keys += list(BLOOM_PRESETS)
    for raw in (args.bloom_presets or "").split(","):
        if raw.strip():
            bloom_keys.append(raw.strip())
    if args.bloom_preset:
        bloom_keys.append(args.bloom_preset)

    enc_keys = list(dict.fromkeys(enc_keys))
    bloom_keys = list(dict.fromkeys(bloom_keys))
    unknown = [k for k in enc_keys if k not in ENCODER_PRESETS]
    if unknown:
        raise SystemExit(
            f"unknown encoder preset(s) {unknown}; choose from {sorted(ENCODER_PRESETS)}"
        )
    unknown = [k for k in bloom_keys if k not in BLOOM_PRESETS]
    if unknown:
        raise SystemExit(
            f"unknown bloom preset(s) {unknown}; choose from {sorted(BLOOM_PRESETS)}"
        )

    if not enc_keys and not bloom_keys:
        return [(PRIMARY_ENCODER, PRIMARY_BLOOM)]
    arms = [(key, PRIMARY_BLOOM) for key in enc_keys]
    if PRIMARY_BLOOM in bloom_keys and not enc_keys:
        arms.append((PRIMARY_ENCODER, PRIMARY_BLOOM))
    arms += [(PRIMARY_ENCODER, key) for key in bloom_keys if key != PRIMARY_BLOOM]
    return list(dict.fromkeys(arms))


def resolve_encoder_specs(args) -> list[EncoderSpec]:
    """Selected arms, each with its endpoints resolved. No network yet.

    Arms come from :func:`resolve_arms`; every one is a preset pair, so its
    model, suffix and env slot always agree.

    Endpoint precedence per arm (slot ``S = encoder_slot(key)``):
      1. --embed-base-urls-for S=<csv>
      2. <S>_EMBED_BASE_URLS          (written by 05_sync_embed_pod_env.py)
      3. --embed-cache-only: a syntactically valid URL that is never called.

    There is deliberately no shared/global endpoint variable: one URL cannot be
    the right endpoint for several arms, and silently scoring bge-m3 against the
    Nemotron pod is exactly the cache-poisoning served_model() exists to catch.
    Weights revisions are per arm for the same reason (--embed-revision-for /
    <S>_EMBED_REVISION).
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

    arms = resolve_arms(args)

    specs: list[EncoderSpec] = []
    for key, bloom_key in arms:
        kind, model, suffix = ENCODER_PRESETS[key]
        bloom_kind, bloom_default_path, bloom_suffix = BLOOM_PRESETS[bloom_key]
        slot = encoder_slot(key)
        label = _tfidf_label(model, ngram) if kind == "tfidf" else model
        # Revision is PER ARM: one revision string cannot pin the weights of
        # several different models, so there is no global form of this. tfidf is
        # pure code in this repo — an HF weights revision would be a lie in the
        # artifact. Precedence mirrors the endpoint one.
        revision = (
            rev_overrides.get(slot)
            or os.environ.get(f"{slot}_EMBED_REVISION")
        ) if kind != "tfidf" else None
        bloom_path = (
            args.bloom_checkpoint
            or os.environ.get("BLOOM_CHECKPOINT")
            or bloom_default_path
        ) if bloom_kind != "lexicon" else None
        spec = EncoderSpec(key=key, kind=kind, model=model,
                           encoder_suffix=suffix, label=label, revision=revision,
                           bloom_key=bloom_key, bloom_kind=bloom_kind,
                           bloom_path=bloom_path, bloom_suffix=bloom_suffix)

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
        if spec.bloom_kind != "lexicon":
            # Same availability semantics as an encoder endpoint: a missing
            # checkpoint skips the whole arm rather than silently falling back
            # to the lexicon, which would label a lexicon-scored artifact as
            # the transformer arm and turn the sensitivity axis into a mirror.
            resolved = Path(spec.bloom_path or "")
            if not resolved.is_absolute():
                resolved = REPO_ROOT / resolved
            if resolved.is_dir():
                spec.bloom_path = str(resolved)
            else:
                spec.available = False
                spec.skip_reason = (
                    f"no Bloom checkpoint at {resolved} (train one with "
                    "evaluator/scripts/train_bloom_6way_classifier.py, or point at "
                    "it with --bloom-checkpoint / BLOOM_CHECKPOINT)"
                )
        if kind != "api":
            specs.append(spec)          # local/tfidf need no endpoint
            continue

        raw = overrides.get(slot) or os.environ.get(f"{slot}_EMBED_BASE_URLS")
        if not raw and args.embed_cache_only:
            raw = "http://embed-cache-only.invalid/v1"
        if raw:
            spec.base_urls = [u.strip() for u in raw.split(",") if u.strip()]
        if not spec.base_urls:
            spec.available = False
            spec.skip_reason = (
                f"no endpoint (run 05_sync_embed_pod_env.py to set "
                f"{slot}_EMBED_BASE_URLS, pass --embed-base-urls-for {slot}=..., "
                f"or use --embed-cache-only)"
            )
        specs.append(spec)

    return specs


def probe_specs(specs: list[EncoderSpec], api_key_of, cache_only: bool) -> None:
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

    An unreachable endpoint marks the arm unavailable and the run goes on
    without it — arms are independent. With a single arm selected that leaves
    nothing runnable, and main() exits rather than scoring against a pod whose
    identity it could not confirm.
    """
    for spec in specs:
        if spec.kind != "api" or not spec.available or cache_only or not spec.base_urls:
            continue
        for url in spec.base_urls:
            live = served_model(url, api_key_of(spec))
            if live is None:
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


#: The reported panel, in reporting order. Derived from the evaluator's
#: PANEL_SIGNALS rather than restated here, so a signal can never be scored by
#: the evaluator and silently dropped by the scorer (or vice versa).
COMPONENTS = [signal for signal, _family in PANEL_SIGNALS]


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
    if payload.get("bloom_classifier") != spec.bloom_key:
        return None, (
            "bloom classifier mismatch: "
            f"{payload.get('bloom_classifier')!r} != {spec.bloom_key!r}"
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
            e["details"]["sanity"]["bloom_vs_declared_agreement"]
            for e in entries
            if e.get("details", {}).get("sanity", {}).get("bloom_vs_declared_agreement")
            is not None
        ]
        means["sanity_agreement"] = (
            statistics.fmean(sanity_values) if sanity_values else None
        )
        rows.append((agent_id, len(entries), means))
    # Console ordering only — sorting the table by one signal is a display
    # choice, not a claim that this signal ranks the agents.
    rows.sort(key=lambda r: -(
        r[2]["objective_assessment_similarity"]
        if r[2]["objective_assessment_similarity"] is not None else -1
    ))
    return rows


#: (COMPONENTS key or extra, console column label) — one entry per column.
#: Family A first, then family B, then the validity-evidence extra.
_TABLE_COLUMNS = [
    ("objective_assessment_similarity", "AsmSim"),
    ("objective_activity_similarity", "ActSim"),
    ("activity_assessment_similarity", "ActAsmSim"),
    # Still emitted, no longer panel endpoints.
    ("objective_evaluation_similarity", "EvlSim"),
    ("assessment_objective_similarity", "RevSim"),
    ("objective_cognitive_congruence", "CogCon"),
    ("porter_mean", "Porter"),
    ("webb_bloom_consistency", "BloomC"),
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


def default_sweep_csv() -> Path:
    """``results/alignment_sweep_<timestamp>.csv``.

    Timestamped rather than fixed because the sweep is normally run in several
    invocations — the served primary arm, then the local arms, then a Bloom
    re-score off the cache — and a fixed name means each one silently replaces
    the previous one's roster. Same ``%Y%m%d_%H%M%S`` stamp the run dirs use.
    """
    from datetime import datetime  # noqa: PLC0415 - only needed on this path

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = REPO_ROOT / "results" / f"alignment_sweep_{stamp}.csv"
    # Two invocations started within the same second — e.g. the served primary
    # arm and the local arms launched side by side — would otherwise land on
    # one name and the second would clobber the first.
    path, n = base, 2
    while path.exists():
        path = base.with_name(f"{base.stem}-{n}.csv")
        n += 1
    return path


def write_sweep_csv(path: Path, rows: list[dict]) -> None:
    """Long-format roster of the whole sweep: one row per (arm, run dir, agent).

    Nothing here is recomputed — these are the same per-agent means the
    per-run tables already show, just stacked with the arm as a column so the
    sweep is inspectable in one file.
    """
    text_cols = ("arm", "encoder", "bloom", "model", "suffix", "run_dir",
                 "agent_id", "n")
    cols = list(text_cols) + COMPONENTS + ["sanity_agreement"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for row in rows:
            writer.writerow([
                row.get(c) if c in text_cols
                else (f"{row[c]:.4f}" if row.get(c) is not None else "")
                for c in cols
            ])
    print(f"\nWrote {path} ({len(rows)} rows)")


def print_sweep_matrix(specs: list[EncoderSpec], scored: list[EncoderSpec],
                       rows: list[dict], n_run_dirs: int) -> None:
    """One line per selected arm: what ran, what didn't, and why.

    Keyed by ``arm_label``, not by encoder: two arms can share an encoder and
    differ only on the Bloom axis.
    """
    focus: dict[str, tuple[float | None, int]] = {}
    for spec in scored:
        vals = [r for r in rows
                if r["arm"] == spec.arm_label and r["agent_id"] == SWEEP_FOCUS_AGENT]
        means = [r["objective_assessment_similarity"] for r in vals
                 if r.get("objective_assessment_similarity") is not None]
        focus[spec.arm_label] = (
            statistics.fmean(means) if means else None,
            sum(r["n"] for r in vals),
        )
    print(f"\n=== sensitivity sweep summary ({len(specs)} arms selected, "
          f"{len(scored)} scored, {len(specs) - len(scored)} skipped)")
    header = (f"{'arm':22s} {'model':44s} {'suffix':10s} {'status':10s} "
              f"{'runs':>5s} {'scen':>6s} {SWEEP_FOCUS_AGENT} AsmSim")
    print(header)
    print("-" * len(header))
    for spec in specs:
        model = spec.model if len(spec.model) <= 44 else spec.model[:41] + "..."
        suffix = spec.suffix or "(primary)"
        if spec in scored:
            mean, n_scen = focus[spec.arm_label]
            cell = f"{mean:.3f}" if mean is not None else "-"
            print(f"{spec.arm_label:22s} {model:44s} {suffix:10s} {'scored':10s} "
                  f"{n_run_dirs:5d} {n_scen:6d} {cell}")
        else:
            print(f"{spec.arm_label:22s} {model:44s} {suffix:10s} {'SKIPPED':10s} "
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
            slot = encoder_slot(spec.key)
            raise SystemExit(
                f"encoder [{spec.key}] needs an endpoint: set {slot}_EMBED_BASE_URLS "
                f"(05_sync_embed_pod_env.py) or pass --embed-base-urls-for {slot}=..., "
                f"e.g. a RunPod pod running 'vllm serve {spec.model} --task embed'. "
                "Pass --embed-cache-only to run purely off the embedding cache."
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
    if spec.kind == "local":
        model_slug = re.sub(r"[^A-Za-z0-9.-]+", "-", spec.model)
        device = args.local_device or _auto_device()
        print(f"  (local encoder on {device}; first run downloads the weights)")
        return SentenceTransformerEncoder(
            model_name=spec.model,
            cache_path=run_dir.parent / f".embed_cache_{model_slug}.sqlite",
            cache_only=args.embed_cache_only,
            revision=spec.revision,
            device=device,
        )
    return TfidfCharNgramEncoder(ngram_range=parse_ngram_range(args.tfidf_ngram_range))


def _auto_device() -> str:
    """Best local device for sentence-transformers, without importing torch eagerly."""
    try:
        import torch  # noqa: PLC0415 - optional heavy dep

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def build_bloom(spec: EncoderSpec, args=None):
    """Instantiate the Bloom classifier for one arm.

    No implicit fallback: an unusable checkpoint has already marked the arm
    unavailable in :func:`resolve_encoder_specs`, because silently scoring the
    transformer arm with the lexicon would make the sensitivity axis agree with
    itself and report that as instrument independence.
    """
    if spec.bloom_kind == "lexicon":
        return LexiconBloomClassifier()
    classifier = TransformerBloomClassifier(
        spec.bloom_path, device=(args.local_device if args else None))
    print(f"  (Bloom arm: {spec.bloom_key} <- {spec.bloom_path} on "
          f"{classifier.device})")
    return classifier


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

    evaluator = AlignmentEvaluator(encoder=encoder,
                                   bloom_classifier=build_bloom(spec, args))

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
                "bloom_classifier": spec.bloom_key,
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
    print("Note: 'Sanity' is the validation-evidence Bloom-vs-declared "
          "agreement; it never enters scoring.")
    return rows




def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score constructive alignment (deterministic, LLM-free) over a run dir."
    )
    parser.add_argument("run_dirs", nargs="+", metavar="RUN_DIR",
                        help="One or more run dirs (globs expand in the shell).")
    parser.add_argument(
        "--encoder-preset", choices=sorted(ENCODER_PRESETS),
        help="Score with one of the sensitivity-design encoders. Alias for a "
             f"one-element --encoder-presets (default: {PRIMARY_ENCODER}).",
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
        "--bloom-preset", choices=sorted(BLOOM_PRESETS),
        help="Score with one of the Bloom-axis arms. Alias for a one-element "
             f"--bloom-presets (default: {PRIMARY_BLOOM}).",
    )
    parser.add_argument(
        "--bloom-presets", default=None, metavar="K1,K2,...",
        help="Bloom classifiers to sweep, e.g. "
             f"'{','.join(BLOOM_PRESETS)}'. This is family B's independence "
             "check: 3 of the 6 panel signals are Bloom-derived, so the lexicon "
             "needs the same kind of axis the encoder has. Non-primary Bloom "
             "arms always pair with the primary encoder (star design), and cost "
             "nearly nothing — Bloom runs on CPU and the vectors come from the "
             "embedding cache, so no pod is needed with --embed-cache-only.",
    )
    parser.add_argument(
        "--bloom-checkpoint", default=None, metavar="PATH",
        help="Checkpoint dir for the transformer Bloom arm, overriding the "
             "preset default (env: BLOOM_CHECKPOINT). Train one with "
             "evaluator/scripts/train_bloom_6way_classifier.py.",
    )
    parser.add_argument(
        "--all-arms", action="store_true",
        help="The whole sensitivity star: every encoder preset (with the "
             "primary Bloom arm) plus every non-primary Bloom arm (with the "
             "primary encoder). Not a grid — one deviation per arm.",
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
        "--sweep-csv", default=None, metavar="PATH",
        help="Long-format roster of the whole sweep (one row per arm x run dir "
             "x agent). Default: results/alignment_sweep_<timestamp>.csv — "
             "timestamped so successive invocations (primary arm, then the "
             "local arms, then a Bloom re-score) accumulate side by side "
             "instead of the last one silently overwriting the rest. Pass an "
             "explicit path to pin it. The roster is ALWAYS written: every "
             "invocation leaves a record of which arms it covered.",
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
        "--local-device", default=None, choices=["mps", "cuda", "cpu"],
        help="Device for everything that runs in-process — 'local'-kind "
             "encoder arms and the transformer Bloom arm (default: auto — mps "
             "on Apple silicon, cuda if present, else cpu).",
    )
    parser.add_argument(
        "--embed-cache-only", action="store_true",
        help="Fail instead of calling the endpoint when a text is missing from "
             "the embedding cache. Lets a re-score run with no pod up, as long "
             "as every text was already embedded by an earlier pass.",
    )
    parser.add_argument(
        "--embed-revision-for", action="append", default=[], metavar="SLOT=REVISION",
        help="Per-arm weights revision (HF commit hash / tag), repeatable, e.g. "
             "--embed-revision-for BGEM3=5617a9f61b028005a4858fdac845db406aefb181. "
             "Falls back to <SLOT>_EMBED_REVISION in the env. Recorded as "
             "'encoder_revision' AND folded into the embedding-cache key, so a "
             "revision bump re-embeds instead of reusing the old weights' "
             "vectors — the model name alone does not pin weights.",
    )
    parser.add_argument(
        "--embed-api-key-env", default="VLLM_API_KEY",
        help="NAME of the env var holding the endpoint's API key (indirection, "
             "same pattern as the ladder's AGENT_MODEL_API_KEY_ENVS). Per-arm "
             "<SLOT>_EMBED_API_KEY_ENV wins over this default.",
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

    specs = resolve_encoder_specs(args)

    # Patterns are expanded here as well as by the shell: quoting a glob is the
    # obvious thing to type (07's --auto-glob requires it), and it used to fail
    # with a confusing "Not a directory: results/..._r[123]_*".
    run_dirs, unmatched = [], []
    for pattern in args.run_dirs:
        hits = sorted(globmod.glob(pattern))
        if hits:
            run_dirs += [Path(h).resolve() for h in hits]
        else:
            unmatched.append(pattern)
    if unmatched:
        raise SystemExit("No run dir matched: " + ", ".join(unmatched))
    not_dirs = [d for d in run_dirs if not d.is_dir()]
    if not_dirs:
        raise SystemExit("Not a directory: " + ", ".join(str(d) for d in not_dirs))
    run_dirs = list(dict.fromkeys(run_dirs))

    # Confirm every endpoint really serves what we are about to label its
    # artifacts with — once per invocation, not per run dir.
    if not args.reaggregate:
        probe_specs(specs, lambda s: os.environ.get(s.api_key_env),
                    args.embed_cache_only)
    if args.require_all:
        blocked = [s for s in specs if not s.available]
        if blocked:
            raise SystemExit("--require-all: " + "; ".join(
                f"[{s.arm_label}] {s.skip_reason}" for s in blocked))

    if not any(s.available for s in specs):
        for spec in specs:
            print(f"  - {spec.arm_label}: SKIPPED — {spec.skip_reason}", file=sys.stderr)
        raise SystemExit("No sweep arm is runnable.")

    # One confirmation for the whole invocation, naming the arm(s) that will
    # overwrite the primary artifact pooling reads.
    writes_primary = [s.arm_label for s in specs if s.available and not s.suffix]
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
            print(f"\n=== arm [{spec.arm_label}] SKIPPED — {spec.skip_reason}")
            continue
        rev = f" @{spec.revision}" if spec.revision else " (revision unpinned)"
        print(f"\n=== arm [{spec.arm_label}] {spec.label}{rev} -> {spec.scores_name}")
        base_encoder = None if args.reaggregate else build_encoder(spec, run_dirs[0], args)
        for i, run_dir in enumerate(run_dirs, 1):
            if len(run_dirs) > 1:
                print(f"\n--- [{i}/{len(run_dirs)}] {run_dir.name}")
            for agent_id, n, means in score_run(run_dir, args, spec, base_encoder):
                sweep_rows.append({"arm": spec.arm_label, "encoder": spec.key,
                                   "bloom": spec.bloom_key, "model": spec.model,
                                   "suffix": spec.suffix, "run_dir": run_dir.name,
                                   "agent_id": agent_id, "n": n, **means})
        scored.append(spec)

    # Unconditional: an invocation that scored nothing still leaves a
    # header-only roster, which is a truthful record of "this run covered no
    # arm" — the failure mode worth avoiding is a sweep whose coverage nobody
    # can reconstruct afterwards.
    write_sweep_csv(Path(args.sweep_csv or default_sweep_csv()), sweep_rows)
    print_sweep_matrix(specs, scored, sweep_rows, len(run_dirs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
