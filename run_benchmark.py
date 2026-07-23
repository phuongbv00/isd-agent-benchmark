#!/usr/bin/env python3
# Warning filter (suppress Python 3.14 + Pydantic V1 compatibility warnings)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pydantic")

"""
ISD Agent Benchmark integration test script

Runs multiple ISD Agents across multiple scenarios and compares them.
The LLM backend is injected via the shared.llm configuration.
"""

import json
import os
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Any, Dict

from shared.llm import LLMConfig, llm_config_from_env

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None

# Add agent module paths
_SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPT_DIR / "agents" / "baseline" / "src"))
sys.path.insert(0, str(_SCRIPT_DIR / "agents" / "eduplanner" / "src"))
sys.path.insert(0, str(_SCRIPT_DIR / "agents" / "react-isd" / "src"))
sys.path.insert(0, str(_SCRIPT_DIR / "agents" / "addie-agent" / "src"))
sys.path.insert(0, str(_SCRIPT_DIR / "agents" / "dick-carey-agent" / "src"))
sys.path.insert(0, str(_SCRIPT_DIR / "agents" / "rpisd-agent" / "src"))


class BenchmarkProgressLogger:
    """Friendly logger that prints benchmark progress."""

    # Per-scenario pipeline steps (shown as a checklist)
    PIPELINE_STEPS = ["Run agents", "Evaluate", "Done"]

    def __init__(self, total_scenarios: int, total_agents: int, log_file: Optional[Path] = None):
        self.total_scenarios = total_scenarios
        self.total_agents = total_agents
        self.total_tasks = total_scenarios * total_agents  # total work units

        self.completed_scenarios = 0
        self.completed_tasks = 0
        self.start_time = time.time()
        self.scenario_times = []  # per-scenario elapsed times

        self.log_file = log_file
        self.lock = threading.Lock()

        if self.log_file:
            self._write_log_header()

    def _append_log(self, text: str):
        """Append a line to the log file (caller holds the lock)."""
        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(text)

    def _write_log_header(self):
        """Write the log file header."""
        header = f"""
================================================================================
  ISD Agent Benchmark run log
  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================

Total scenarios: {self.total_scenarios}
Total agents: {self.total_agents}
Total tasks: {self.total_tasks} (scenarios x agents)

================================================================================
"""
        if self.log_file:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write(header)
        print(header)

    def _estimate_remaining_time(self) -> str:
        """Estimate the remaining time."""
        if not self.scenario_times:
            return "calculating..."

        avg_time = sum(self.scenario_times) / len(self.scenario_times)
        remaining_scenarios = self.total_scenarios - self.completed_scenarios
        remaining_seconds = avg_time * remaining_scenarios

        if remaining_seconds < 60:
            return f"{int(remaining_seconds)}s"
        elif remaining_seconds < 3600:
            return f"{int(remaining_seconds / 60)}m {int(remaining_seconds % 60)}s"
        else:
            hours = int(remaining_seconds / 3600)
            minutes = int((remaining_seconds % 3600) / 60)
            return f"{hours}h {minutes}m"

    def _estimate_completion_time(self) -> str:
        """Estimate the completion time."""
        if not self.scenario_times:
            return "calculating..."

        avg_time = sum(self.scenario_times) / len(self.scenario_times)
        remaining_scenarios = self.total_scenarios - self.completed_scenarios
        remaining_seconds = avg_time * remaining_scenarios

        completion_time = datetime.now() + timedelta(seconds=remaining_seconds)
        return completion_time.strftime('%H:%M:%S')

    def _get_progress_bar(self, current: int, total: int, width: int = 30) -> str:
        """Build a progress bar."""
        if total == 0:
            return "░" * width

        filled = int(width * current / total)
        empty = width - filled
        percentage = (current / total) * 100

        return f"{'█' * filled}{'░' * empty} {percentage:5.1f}%"

    def _format_elapsed_time(self) -> str:
        """Format the elapsed time."""
        elapsed = time.time() - self.start_time
        if elapsed < 60:
            return f"{int(elapsed)}s"
        elif elapsed < 3600:
            return f"{int(elapsed / 60)}m {int(elapsed % 60)}s"
        else:
            hours = int(elapsed / 3600)
            minutes = int((elapsed % 3600) / 60)
            return f"{hours}h {minutes}m"


    def log_step(self, scenario_id: str, step_index: int, status: str):
        """Log a per-scenario pipeline step.

        status in {"start", "done", "skip"}.
        """
        total_steps = len(self.PIPELINE_STEPS)
        label = self.PIPELINE_STEPS[step_index]
        icon = {"start": "▶", "done": "✅", "skip": "⏭"}.get(status, "▶")
        with self.lock:
            print(f"[{scenario_id}] [{step_index + 1}/{total_steps}] {icon} {label}")
            self._append_log(
                f"[{datetime.now().strftime('%H:%M:%S')}] [{scenario_id}] "
                f"step {step_index + 1}/{total_steps} {label}: {status}\n"
            )

    def log_eval_progress(self, scenario_id: str, done: int, total: int,
                          agent_id: str = "", judge_model: str = ""):
        """Log evaluate-phase progress as a percentage of judge-calls.

        ``judge_model`` is the judge model name (e.g. ``google/gemini-2.5-flash-lite``).
        """
        with self.lock:
            bar = self._get_progress_bar(done, total, width=20)
            detail = f" – {agent_id} · {judge_model}" if agent_id else ""
            print(f"[{scenario_id}]   Evaluate [{bar}] ({done}/{total} judge calls){detail}")
            self._append_log(
                f"[{datetime.now().strftime('%H:%M:%S')}] [{scenario_id}] "
                f"evaluate {done}/{total} judge calls{detail}\n"
            )

    def log_scenario_start(self, scenario_id: str, scenario_index: int):
        """Log the start of a scenario."""
        with self.lock:
            progress_bar = self._get_progress_bar(scenario_index, self.total_scenarios)
            remaining = self._estimate_remaining_time()
            completion = self._estimate_completion_time()

            log_msg = f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│ 📊 Progress: [{progress_bar}]
│
│ 🔄 Current: [{scenario_index + 1}/{self.total_scenarios}] {scenario_id}
│ ⏱️  Elapsed: {self._format_elapsed_time()}
│ ⏳ Est. remaining: {remaining}
│ 🏁 Est. completion: {completion}
│
│ Scenarios left: {self.total_scenarios - scenario_index - 1}
└─────────────────────────────────────────────────────────────────────────────┘
"""
            print(log_msg)
            self._append_log(
                f"\n[{datetime.now().strftime('%H:%M:%S')}] Start: {scenario_id} "
                f"({scenario_index + 1}/{self.total_scenarios})\n"
            )

    def log_agent_progress(self, scenario_id: str, agent_id: str, status: str):
        """Log per-agent progress within a scenario.

        Agents run concurrently, so a positional index would be misleading;
        only the agent name and its status are shown.
        """
        with self.lock:
            status_icon = "✅" if status == "success" else "❌" if status == "failed" else "🔄"
            print(f"[{scenario_id}]     {status_icon} {agent_id}: {status}")
            self._append_log(f"  - {agent_id}: {status}\n")

    def log_scenario_complete(self, scenario_id: str, elapsed_seconds: float, success_count: int):
        """Log the completion of a scenario."""
        with self.lock:
            self.completed_scenarios += 1
            self.scenario_times.append(elapsed_seconds)

            # Running count of completed scenarios (accurate even under parallelism)
            done = self.completed_scenarios
            total = self.total_scenarios
            overall_bar = self._get_progress_bar(done, total, width=20)
            remaining = self._estimate_remaining_time()

            log_msg = f"""
    ────────────────────────────────────────────────────────────
    ✅ Done: {scenario_id}
    ⏱️ Elapsed: {elapsed_seconds:.1f}s
    📈 Successful agents: {success_count}/{self.total_agents}
    📊 Scenarios completed: [{overall_bar}] {done}/{total}  (est. remaining: {remaining})
    ────────────────────────────────────────────────────────────
"""
            print(log_msg)

            self._append_log(
                f"[{datetime.now().strftime('%H:%M:%S')}] Done: {scenario_id} "
                f"({elapsed_seconds:.1f}s, success: {success_count}) "
                f"[{done}/{total} scenarios completed]\n"
            )

    def log_final_summary(self, results: dict):
        """Log the final summary."""
        total_elapsed = time.time() - self.start_time

        total_success = 0
        total_failed = 0
        for variant_results in results.get("scenarios", {}).values():
            for scenario_result in variant_results.values():
                for agent_result in scenario_result.get("agents", {}).values():
                    if agent_result.get("success"):
                        total_success += 1
                    else:
                        total_failed += 1

        summary = f"""

================================================================================
  🏁 Benchmark complete!
================================================================================

📊 Final results
────────────────────────────────────────────────────────────────────────────────
  Total scenarios:  {self.completed_scenarios}
  Total tasks:      {total_success + total_failed}
  Success:          {total_success} ✅
  Failed:           {total_failed} ❌
  Success rate:     {(total_success / (total_success + total_failed) * 100) if (total_success + total_failed) > 0 else 0:.1f}%

⏱️ Total elapsed: {self._format_elapsed_time()}
📁 Results saved at: {results.get('output_dir', 'N/A')}

================================================================================
"""
        print(summary)

        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(summary)

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  # Skip if dotenv is not available


# Project root path (isd-agent-bench-en directory)
PROJECT_ROOT = Path(__file__).parent
SCENARIOS_DIR = PROJECT_ROOT / "scenarios"
RESULTS_DIR = PROJECT_ROOT / "results"
VENV_BIN = PROJECT_ROOT / ".venv" / "bin"

# Environment setup (for running agents)
def get_env_with_venv(extra_env: Optional[dict[str, str]] = None):
    """Return environment variables with the venv bin path prepended."""
    env = os.environ.copy()
    venv_path = str(VENV_BIN)
    current_path = env.get('PATH', '')
    env['PATH'] = f"{venv_path}:{current_path}"
    if extra_env:
        env.update({k: v for k, v in extra_env.items() if v is not None})
    return env


def get_all_scenarios(
    use_stratified_sampling: bool = False,
    n_samples: int | None = None,
    sampling_strategy: str = "oversample",
    dataset: str | None = None,
) -> dict[str, list[Path]]:
    """
    Collect all scenario files per variant (IDLD dataset structure).

    Dataset structure:
    - dataset=None (default): use existing variant directories (idld_aligned, context_variant)
    - dataset="<name>": load scenarios/<name>/ directly, e.g. train, test, test_90

    Args:
        use_stratified_sampling: Whether to use stratified sampling (correct imbalanced axes)
        n_samples: Number of scenarios to sample (None means all)
        sampling_strategy: Sampling strategy ("oversample", "undersample", "proportional")
        dataset: Dataset directory name under scenarios/ (None=variant mode)

    Returns:
        Dictionary of scenario file paths per variant/dataset
    """
    # dataset mode: load directly from a named scenarios/<dataset> directory
    if dataset:
        dataset_dir = SCENARIOS_DIR / dataset
        if not dataset_dir.exists():
            print(f"[warning] {dataset} directory does not exist: {dataset_dir}")
            return {dataset: []}
        if not dataset_dir.is_dir():
            print(f"[warning] {dataset} is not a directory: {dataset_dir}")
            return {dataset: []}

        scenario_files = sorted(dataset_dir.glob("*.json"))
        print(f"  [{dataset.upper()}] loaded {len(scenario_files)} scenarios")

        # Stratified sampling support (only meaningful for the train dataset)
        if use_stratified_sampling and n_samples and dataset == "train":
            try:
                from scenarios.sampling_strategy import StratifiedScenarioSampler
                sampler = StratifiedScenarioSampler(scenarios_dir=dataset_dir)
                sampled = sampler.sample_with_paths(n_samples, strategy=sampling_strategy)
                scenario_files = [path for path, _ in sampled]
                print(f"  [stratified sampling] {dataset}: selected {len(scenario_files)} (strategy: {sampling_strategy})")
            except ImportError:
                pass  # Use all if the sampling module is unavailable

        return {dataset: scenario_files}

    # Existing variant mode: use idld_aligned, context_variant directories
    scenarios = {"idld_aligned": [], "context_variant": []}

    for variant in scenarios.keys():
        variant_dir = SCENARIOS_DIR / variant
        if variant_dir.exists():
            if use_stratified_sampling and variant == "idld_aligned" and n_samples:
                # Apply stratified sampling (correct imbalanced axes)
                try:
                    from scenarios.sampling_strategy import StratifiedScenarioSampler
                    sampler = StratifiedScenarioSampler(scenarios_dir=variant_dir)
                    sampled = sampler.sample_with_paths(n_samples, strategy=sampling_strategy)
                    scenarios[variant] = [path for path, _ in sampled]
                    print(f"  [stratified sampling] {variant}: selected {len(scenarios[variant])} (strategy: {sampling_strategy})")
                except ImportError:
                    # Default behavior if the sampling module is unavailable
                    for scenario_file in sorted(variant_dir.glob("*.json")):
                        scenarios[variant].append(scenario_file)
            else:
                for scenario_file in sorted(variant_dir.glob("*.json")):
                    scenarios[variant].append(scenario_file)

    return scenarios


def install_agents() -> bool:
    """Install all agent packages."""
    print("\n" + "=" * 60)
    print("Installing agent packages")
    print("=" * 60)

    agents = [
        PROJECT_ROOT / "agents" / "eduplanner",
        PROJECT_ROOT / "agents" / "baseline",
        PROJECT_ROOT / "agents" / "react-isd",
        PROJECT_ROOT / "agents" / "addie-agent",
        PROJECT_ROOT / "agents" / "dick-carey-agent",
        PROJECT_ROOT / "agents" / "rpisd-agent",
        PROJECT_ROOT / "agents" / "alignmentgraph-isd",
        PROJECT_ROOT / "evaluator",
    ]

    for agent_path in agents:
        print(f"\nInstalling: {agent_path.name}")
        result = subprocess.run(
            ["pip", "install", "-e", str(agent_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"  Error: {result.stderr}")
            return False
        print(f"  Done")

    return True


def check_agents_installed() -> dict[str, bool]:
    """Check whether agent modules can be imported."""
    agents = {
        "eduplanner": False,
        "baseline": False,
        "react-isd": False,
        "addie-agent": False,
        "dick-carey-agent": False,
        "rpisd-agent": False,
        "alignmentgraph-isd": False,
    }

    # Module import test
    try:
        from baseline.generator import BaselineGenerator
        agents["baseline"] = True
    except ImportError:
        pass

    try:
        from eduplanner.agents import EduPlannerAgent
        agents["eduplanner"] = True
    except ImportError:
        pass

    try:
        from react_isd.agent import ReActISDAgent
        agents["react-isd"] = True
    except ImportError:
        pass

    try:
        from alignmentgraph_isd_agent import AlignmentGraphISDAgent
        agents["alignmentgraph-isd"] = True
    except ImportError:
        pass

    try:
        from addie_agent.agent import ADDIEAgent
        agents["addie-agent"] = True
    except ImportError:
        pass

    try:
        from dick_carey_agent.agent import DickCareyAgent
        agents["dick-carey-agent"] = True
    except ImportError:
        pass

    try:
        from rpisd_agent.agent import RPISDAgent
        agents["rpisd-agent"] = True
    except ImportError:
        pass

    return agents


def _resolve_llm_config(llm_config: Optional[LLMConfig] = None) -> LLMConfig:
    return llm_config or llm_config_from_env()


def _normalize_agent_max_tokens(llm_config: LLMConfig) -> LLMConfig:
    """Give every agent the same per-call output budget.

    Respects an explicit ``AGENT_MODEL_MAX_TOKENS`` (already folded into the
    resolved config by ``llm_config_from_env``) — that value is uniform across
    agents too — otherwise pins the uniform benchmark default. This is the single
    place per-agent max_tokens is set; individual runners must not override it."""
    if os.getenv("AGENT_MODEL_MAX_TOKENS"):
        return llm_config
    return llm_config.copy_with(max_tokens=16384)


def _build_judge_env(
    *,
    provider: Optional[str] = None,
    providers: Optional[str] = None,
    models: Optional[str] = None,
    base_url: Optional[str] = None,
    base_urls: Optional[str] = None,
    api_key: Optional[str] = None,
    api_keys: Optional[str] = None,
    api_key_env: Optional[str] = None,
    api_key_envs: Optional[str] = None,
    credential_strategies: Optional[str] = None,
) -> dict[str, str]:
    """Build env overrides consumed by isd-evaluator judge config."""
    env: dict[str, str] = {}
    mapping = {
        "JUDGE_MODEL_PROVIDER": provider,
        "JUDGE_MODEL_PROVIDERS": providers,
        "JUDGE_MODEL_NAMES": models,
        "JUDGE_MODEL_BASE_URL": base_url,
        "JUDGE_MODEL_BASE_URLS": base_urls,
        "JUDGE_MODEL_API_KEY": api_key,
        "JUDGE_MODEL_API_KEYS": api_keys,
        "JUDGE_MODEL_API_KEY_ENV": api_key_env,
        "JUDGE_MODEL_API_KEY_ENVS": api_key_envs,
        "JUDGE_MODEL_CREDENTIAL_STRATEGIES": credential_strategies,
    }
    for key, value in mapping.items():
        if value:
            env[key] = value
    return env


def _judge_models_from_env(judge_env: Optional[dict[str, str]]) -> list[str]:
    models = (judge_env or {}).get("JUDGE_MODEL_NAMES") or os.getenv("JUDGE_MODEL_NAMES")
    if models:
        return [part.strip() for part in models.split(",") if part.strip()]
    return ["openai/gpt-4o-mini", "google/gemini-2.5-flash-lite"]


def _get_agent_runner(agent_id: str, llm_config: Optional[LLMConfig] = None):
    """Return the run function for the given agent ID (module-based)."""
    llm_config = _normalize_agent_max_tokens(_resolve_llm_config(llm_config))
    uniform_max_tokens = llm_config.max_tokens

    if agent_id == "baseline":
        from baseline.generator import BaselineGenerator
        def run_baseline(scenario: dict) -> dict:
            # Pass max_tokens explicitly: BaselineGenerator's __init__ default
            # (32768) would otherwise override the uniform budget on the config.
            gen = BaselineGenerator(llm_config=llm_config, max_tokens=uniform_max_tokens)
            return gen.generate(scenario)
        return run_baseline

    elif agent_id == "eduplanner":
        from eduplanner.agents import EduPlannerAgent
        from eduplanner.agents.base import AgentConfig
        from eduplanner.models.schemas import ScenarioInput
        def run_eduplanner(scenario: dict) -> dict:
            config = AgentConfig(
                model=llm_config.model,
                provider=llm_config.provider,
                llm_config=llm_config,
            )
            agent = EduPlannerAgent(config=config, max_iterations=3, target_score=90.0)
            scenario_input = ScenarioInput(**scenario)
            result = agent.run(scenario_input)
            return {
                "addie_output": result.addie_output.to_standard_dict(),
                "trajectory": result.trajectory.model_dump(),
                "metadata": result.metadata.model_dump(),
            }
        return run_eduplanner

    elif agent_id == "react-isd":
        from react_isd.agent import ReActISDAgent
        def run_react(scenario: dict) -> dict:
            agent = ReActISDAgent(llm_config=llm_config)
            return agent.run(scenario)
        return run_react

    elif agent_id == "alignmentgraph-isd":
        from alignmentgraph_isd_agent import AlignmentGraphISDAgent
        def run_alignmentgraph_isd(scenario: dict) -> dict:
            agent = AlignmentGraphISDAgent(llm_config=llm_config)
            return agent.run(scenario)
        return run_alignmentgraph_isd

    elif agent_id == "addie-agent":
        from addie_agent.agent import ADDIEAgent
        def run_addie(scenario: dict) -> dict:
            agent = ADDIEAgent(llm_config=llm_config)
            return agent.run(scenario)
        return run_addie

    elif agent_id == "dick-carey-agent":
        from dick_carey_agent.agent import DickCareyAgent
        def run_dickcarey(scenario: dict) -> dict:
            agent = DickCareyAgent(llm_config=llm_config)
            return agent.run(scenario)
        return run_dickcarey

    elif agent_id == "rpisd-agent":
        from rpisd_agent.agent import RPISDAgent
        def run_rpisd(scenario: dict) -> dict:
            agent = RPISDAgent(llm_config=llm_config)
            return agent.run(scenario)
        return run_rpisd

    else:
        raise ValueError(f"Unknown agent: {agent_id}")


def _run_agent_task(
    agent_id: str,
    scenario_path: Path,
    output_dir: Path,
    llm_config: Optional[LLMConfig] = None,
    semaphore: Optional[threading.Semaphore] = None,
) -> tuple[str, dict]:
    """Individual agent execution task (module-based, for parallel execution)."""
    if semaphore:
        semaphore.acquire()

    try:
        output_path = output_dir / f"{agent_id}_output.json"
        trajectory_path = output_dir / f"{agent_id}_trajectory.json"
        log_path = output_dir / f"{agent_id}_log.txt"

        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            with open(scenario_path, "r", encoding="utf-8") as f:
                scenario = json.load(f)

            # Run the agent (module-based). Bracket the run with the thread-local
            # token accounter so every agent gets a uniform token_usage in its
            # metadata (baselines included), regardless of whether the agent
            # tracks tokens itself. See shared/llm/token_accounting.py.
            from shared.llm import token_accounting
            token_accounting.reset()
            start_time = time.time()
            runner = _get_agent_runner(agent_id, llm_config=llm_config)
            result = runner(scenario)
            elapsed = time.time() - start_time
            token_usage = token_accounting.snapshot()
            if isinstance(result, dict):
                result.setdefault("metadata", {})
                if isinstance(result["metadata"], dict):
                    result["metadata"]["token_usage"] = token_usage
                    result["metadata"]["execution_time_seconds"] = elapsed

            addie_output = result.get("addie_output", result)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(addie_output, f, ensure_ascii=False, indent=2, default=str)

            trajectory_data = {
                "scenario_id": scenario.get("scenario_id", "unknown"),
                "agent_id": agent_id,
                "timestamp": datetime.now().isoformat(),
                "trajectory": result.get("trajectory", {}),
                "metadata": result.get("metadata", {"execution_time_seconds": elapsed}),
            }
            with open(trajectory_path, "w", encoding="utf-8") as f:
                json.dump(trajectory_data, f, ensure_ascii=False, indent=2, default=str)

            # Save the alignment-graph dump when the agent provides one
            # (its own files, kept separate from output/trajectory).
            if isinstance(result, dict):
                graph_dump = result.get("graph")
                if isinstance(graph_dump, dict) and graph_dump:
                    with open(output_dir / f"{agent_id}_graph.json", "w", encoding="utf-8") as f:
                        json.dump(graph_dump, f, ensure_ascii=False, indent=2, default=str)
                graph_dot = result.get("graph_dot")
                if isinstance(graph_dot, str) and graph_dot:
                    with open(output_dir / f"{agent_id}_graph.dot", "w", encoding="utf-8") as f:
                        f.write(graph_dot)

            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"=== {agent_id} execution log ===\n")
                f.write(f"Scenario: {scenario_path}\n")
                f.write(f"Elapsed: {elapsed:.2f}s\n")
                f.write(f"Status: SUCCESS\n")

            return agent_id, {
                "success": True,
                "output_path": str(output_path),
                "trajectory_path": str(trajectory_path),
                "log_path": str(log_path),
            }

        except Exception as e:
            error_msg = str(e)
            tb = traceback.format_exc()

            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"=== {agent_id} execution log ===\n")
                f.write(f"Scenario: {scenario_path}\n")
                f.write(f"Status: FAILED\n")
                f.write(f"Error: {error_msg}\n\n")
                f.write("=== Traceback ===\n")
                f.write(tb)

            return agent_id, {
                "success": False,
                "error": error_msg[:500],
                "log_path": str(log_path),
            }

    finally:
        if semaphore:
            semaphore.release()


class _EvalResult:
    """Lightweight stand-in for subprocess.CompletedProcess from streaming."""

    def __init__(self, returncode: int, stdout: str, stderr: str):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _run_evaluation_streaming(
    cmd: list[str],
    env: dict[str, str],
    logger: Optional["BenchmarkProgressLogger"],
    scenario_id: str,
) -> _EvalResult:
    """Run the evaluator subprocess, streaming stdout to parse progress markers.

    Lines starting with ``__EVAL_PROGRESS__`` are turned into evaluate-phase
    progress updates; all other stdout lines are captured and returned so the
    caller behaves like the previous ``subprocess.run`` call.
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        # No timeout - LLM evaluation can take a long time.
    )

    # Drain stderr in a background thread so the child never blocks on a full
    # stderr pipe while this parent is busy reading stdout (which would deadlock).
    captured_stderr: list[str] = []

    def _drain_stderr() -> None:
        if proc.stderr is None:
            return
        for err_line in proc.stderr:
            captured_stderr.append(err_line)

    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stderr_thread.start()

    captured_stdout: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        if line.startswith("__EVAL_PROGRESS__"):
            parts = line.rstrip("\n").split("\t")
            # __EVAL_PROGRESS__ <done> <total> <agent_id> <judge_model>
            if len(parts) >= 3 and logger is not None:
                try:
                    done = int(parts[1])
                    total = int(parts[2])
                    agent_id = parts[3] if len(parts) > 3 else ""
                    judge_model = parts[4] if len(parts) > 4 else ""
                    logger.log_eval_progress(scenario_id, done, total, agent_id, judge_model)
                except (ValueError, IndexError):
                    captured_stdout.append(line)
            continue
        captured_stdout.append(line)

    returncode = proc.wait()
    stderr_thread.join()
    return _EvalResult(returncode, "".join(captured_stdout), "".join(captured_stderr))


def run_single_benchmark(
    scenario_path: Path,
    output_dir: Path,
    agents: Optional[list[str]] = None,
    verbose: bool = False,
    parallel: bool = False,
    max_workers: int = 3,
    multi_judge: bool = True,
    llm_config: Optional[LLMConfig] = None,
    judge_env: Optional[dict[str, str]] = None,
    logger: Optional["BenchmarkProgressLogger"] = None,
    scenario_id: Optional[str] = None,
) -> dict:
    """Run the benchmark for a single scenario.

    Args:
        scenario_path: Path to the scenario file
        output_dir: Output directory
        agents: List of agents to run
        verbose: Verbose output
        parallel: Run agents in parallel
        max_workers: Maximum number of concurrent agents
        logger: Optional progress logger for step/agent/eval progress
        scenario_id: Scenario identifier (defaults to the file stem)
    """
    agents = agents or ["eduplanner", "baseline", "react-isd", "addie-agent", "dick-carey-agent", "rpisd-agent"]
    sid = scenario_id or scenario_path.stem

    print(f"\nScenario: {scenario_path.name}")
    print("-" * 40)

    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "scenario": scenario_path.name,
        "agents": {},
        "timestamp": datetime.now().isoformat(),
    }

    # Step 1: Run agents
    if logger:
        logger.log_step(sid, 0, "start")

    if parallel and len(agents) > 1:
        effective_workers = max_workers
        print(f"  [parallel mode] running {len(agents)} agents concurrently (max_workers={effective_workers})")
        semaphore = threading.Semaphore(effective_workers)

        with ThreadPoolExecutor(max_workers=len(agents)) as executor:
            futures = {}
            for agent_id in agents:
                future = executor.submit(
                    _run_agent_task,
                    agent_id,
                    scenario_path,
                    output_dir,
                    llm_config,
                    semaphore,
                )
                futures[future] = agent_id
                # Rate limit handling: delay between agent submissions
                time.sleep(float(os.getenv("BENCHMARK_DELAY", "2.0")))

            for future in as_completed(futures):
                agent_id = futures[future]
                try:
                    _, agent_result = future.result()
                    results["agents"][agent_id] = agent_result
                    if logger:
                        logger.log_agent_progress(
                            sid, agent_id,
                            "success" if agent_result["success"] else "failed",
                        )
                    else:
                        print(f"  {agent_id}: {'Done' if agent_result['success'] else 'Failed'}")

                    if verbose and not agent_result["success"]:
                        print(f"    stderr: {agent_result.get('stderr', '')[:200]}")
                except Exception as e:
                    print(f"  {agent_id}: exception raised")
                    results["agents"][agent_id] = {
                        "success": False,
                        "error": str(e),
                    }
    else:
        # Sequential execution
        for agent_id in agents:
            if not logger:
                print(f"  running {agent_id}...", end=" ", flush=True)
            _, agent_result = _run_agent_task(
                agent_id,
                scenario_path,
                output_dir,
                llm_config=llm_config,
            )

            results["agents"][agent_id] = agent_result
            if logger:
                logger.log_agent_progress(
                    sid, agent_id,
                    "success" if agent_result["success"] else "failed",
                )
            else:
                print("Done" if agent_result["success"] else "Failed")

            if verbose and not agent_result["success"]:
                print(f"    stderr: {agent_result.get('stderr', '')[:200]}")

            # Rate limit handling: delay between agents
            time.sleep(float(os.getenv("BENCHMARK_DELAY", "2.0")))

    if logger:
        logger.log_step(sid, 0, "done")

    # Step 2: Evaluate
    successful_agents = [a for a, r in results["agents"].items() if r.get("success")]

    if len(successful_agents) >= 2:
        if logger:
            logger.log_step(sid, 1, "start")
        else:
            print(f"  Evaluating...", end=" ", flush=True)

        cmd = [
            "isd-evaluator",
            "compare",
            "--scenario", str(scenario_path),
            "--output-dir", str(output_dir),
            "--agents", ",".join(successful_agents),
        ]

        if multi_judge:
            cmd.append("--multi-judge")
        else:
            cmd.extend(["--single-judge", "--use-llm"])

        if verbose:
            cmd.append("--verbose")

        # Enable structured progress markers from the evaluator subprocess.
        eval_env = get_env_with_venv(judge_env)
        eval_env["ISD_EVAL_PROGRESS"] = "1"

        result = _run_evaluation_streaming(cmd, eval_env, logger, sid)

        if result.returncode == 0:
            if logger:
                logger.log_step(sid, 1, "done")
            else:
                print("Done")

            report_path = output_dir / "comparison_report.json"
            if report_path.exists():
                with open(report_path, "r", encoding="utf-8") as f:
                    results["evaluation"] = json.load(f)
        else:
            if logger:
                logger.log_step(sid, 1, "done")
            else:
                print("Failed")
            results["evaluation_error"] = result.stderr[:500] if result.stderr else "Unknown error"
    else:
        if logger:
            logger.log_step(sid, 1, "skip")
        else:
            print(f"  Skipping evaluation (successful agents: {len(successful_agents)})")
        results["evaluation_skipped"] = True

    # Step 3: Done
    if logger:
        logger.log_step(sid, 2, "done")

    return results


def _run_scenario_task(
    scenario_path: Path,
    output_dir: Path,
    agents: Optional[list[str]],
    verbose: bool,
    parallel: bool,
    max_workers: int,
    multi_judge: bool = True,
    llm_config: Optional[LLMConfig] = None,
    judge_env: Optional[dict[str, str]] = None,
    semaphore: Optional[threading.Semaphore] = None,
    logger: Optional["BenchmarkProgressLogger"] = None,
) -> tuple[str, dict]:
    """Scenario execution task (for scenario-level parallelization)"""
    if semaphore:
        semaphore.acquire()

    try:
        scenario_id = scenario_path.stem
        result = run_single_benchmark(
            scenario_path=scenario_path,
            output_dir=output_dir,
            agents=agents,
            verbose=verbose,
            parallel=parallel,
            max_workers=max_workers,
            multi_judge=multi_judge,
            llm_config=llm_config,
            judge_env=judge_env,
            logger=logger,
            scenario_id=scenario_id,
        )
        return scenario_id, result
    finally:
        if semaphore:
            semaphore.release()


def run_full_benchmark(
    variants: Optional[list[str]] = None,
    agents: Optional[list[str]] = None,
    verbose: bool = False,
    parallel: bool = True,
    max_workers: int = 6,
    scenario_parallel: bool = True,
    scenario_max_workers: int = 8,
    dataset: Optional[str] = None,
    multi_judge: bool = True,
    llm_config: Optional[LLMConfig] = None,
    judge_env: Optional[dict[str, str]] = None,
    run_tag: Optional[str] = None,
) -> dict:
    """Run the full benchmark (IDLD dataset structure).

    Args:
        variants: List of variants to run (used only when dataset=None)
        agents: List of agents to run
        verbose: Verbose output
        parallel: Whether to run agents in parallel (default: True)
        max_workers: Number of concurrent agents (default: 6)
        scenario_parallel: Whether to run scenarios in parallel (default: True)
        scenario_max_workers: Number of concurrent scenarios (default: 8)
        dataset: Dataset selection ("train", "test", None=variant mode)
        run_tag: Optional label inserted into the run directory name
            (e.g. "r1" -> test_90_benchmark_<model>_r1_<timestamp>), used to
            distinguish repeated runs of the same model.
    """
    # Ignore variants when in dataset mode
    if dataset:
        variants = [dataset]  # Treat the dataset like a variant
    else:
        variants = variants or ["idld_aligned", "context_variant"]

    agents = agents or ["eduplanner", "baseline", "react-isd", "addie-agent", "dick-carey-agent", "rpisd-agent"]
    llm_config = _resolve_llm_config(llm_config)

    # Collect scenarios first to know the total count
    all_scenarios = get_all_scenarios(dataset=dataset)
    total_scenarios = sum(len(s) for s in all_scenarios.values())

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build a directory-safe model name (e.g. anthropic/claude-opus-4.5 -> claude-opus-4.5)
    model_name = llm_config.model
    model_safe_name = model_name.split("/")[-1].replace(":", "-")  # Remove provider prefix and replace colons

    # Optional run tag (e.g. "r1") to distinguish repeated runs of the same model
    tag_part = f"_{run_tag}" if run_tag else ""

    # Reflect dataset mode in the directory name (e.g. test_benchmark_claude-opus-4.5_20260122_...)
    if dataset:
        run_dir = RESULTS_DIR / f"{dataset}_benchmark_{model_safe_name}{tag_part}_{timestamp}"
    else:
        run_dir = RESULTS_DIR / f"benchmark_{model_safe_name}{tag_part}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    log_file = run_dir / "benchmark_progress.log"

    logger = BenchmarkProgressLogger(
        total_scenarios=total_scenarios,
        total_agents=len(agents),
        log_file=log_file
    )

    print(f"\n{'=' * 80}")
    print(f"  🚀 ISD Agent Benchmark run")
    print(f"{'=' * 80}")
    print(f"  🤖 Model: {model_name} (provider: {llm_config.provider}, api_spec: {llm_config.api_spec})")

    if dataset:
        print(f"  📂 Dataset mode: {dataset.upper()}")
    else:
        print(f"  📂 Variant mode: {', '.join(variants)}")

    print(f"  Agents ({len(agents)}): {', '.join(agents)}")
    print(f"  Total scenarios: {total_scenarios}")
    print(f"  Parallel mode: {scenario_max_workers} scenarios x {max_workers} agents concurrently")
    if multi_judge:
        judge_models = _judge_models_from_env(judge_env)
        print(f"  Evaluation: Multi-Judge ({len(judge_models)} judge models)")
        for judge_model in judge_models:
            print(f"    - {judge_model}")
    else:
        print(f"  Evaluation: Single-Judge")
    print(f"  Output: {run_dir}")
    print(f"  Log file: {log_file}")
    print(f"{'=' * 80}\n")

    results = {
        "timestamp": timestamp,
        "output_dir": str(run_dir),
        "model": {
            "name": model_name,
            "provider": llm_config.provider,
            "api_spec": llm_config.api_spec,
            "base_url": llm_config.base_url,
        },
        "config": {
            "variants": variants,
            "dataset": dataset,
            "agents": agents,
            "run_tag": run_tag,
            "parallel": parallel,
            "max_workers": max_workers,
            "scenario_parallel": scenario_parallel,
            "scenario_max_workers": scenario_max_workers,
            "multi_judge": multi_judge,
            "judge_models": _judge_models_from_env(judge_env) if multi_judge else None,
            "judge_env": {
                key: value
                for key, value in (judge_env or {}).items()
                if "API_KEY" not in key
            } if judge_env else None,
        },
        "scenarios": {},
    }

    scenario_index = 0

    for variant in variants:
        scenarios = all_scenarios.get(variant, [])
        if not scenarios:
            print(f"\n[{variant}] no scenarios")
            continue

        print(f"\n{'─' * 80}")
        print(f"  📁 [{variant.upper()}] starting {len(scenarios)} scenarios")
        print(f"{'─' * 80}")

        results["scenarios"][variant] = {}

        scenario_iter = scenarios
        if TQDM_AVAILABLE:
            scenario_iter = tqdm(
                scenarios,
                desc=f"[{variant}]",
                unit="scenario",
                leave=True,
                ncols=100,
            )

        if scenario_parallel and len(scenarios) > 1:
            # Scenario-level parallel execution (rate limit handling: cap worker count)
            effective_workers = scenario_max_workers
            semaphore = threading.Semaphore(effective_workers)
            completed_count = 0
            completed_lock = threading.Lock()
            pbar = scenario_iter if TQDM_AVAILABLE else None

            def run_with_logging(scenario_path, scenario_output_dir, idx):
                nonlocal completed_count
                scenario_id = scenario_path.stem
                start_time = time.time()

                # Rate limit handling: delay before starting
                time.sleep(float(os.getenv("BENCHMARK_DELAY", "2.0")))

                logger.log_scenario_start(scenario_id, idx)

                result = run_single_benchmark(
                    scenario_path=scenario_path,
                    output_dir=scenario_output_dir,
                    agents=agents,
                    verbose=verbose,
                    parallel=parallel,
                    max_workers=max_workers,
                    multi_judge=multi_judge,
                    llm_config=llm_config,
                    judge_env=judge_env,
                    logger=logger,
                    scenario_id=scenario_id,
                )

                elapsed = time.time() - start_time
                success_count = sum(1 for r in result.get("agents", {}).values() if r.get("success"))
                logger.log_scenario_complete(scenario_id, elapsed, success_count)

                with completed_lock:
                    completed_count += 1
                    if pbar:
                        pbar.update(1)

                return scenario_id, result

            with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                futures = {}
                for idx, scenario_path in enumerate(scenarios):
                    scenario_output_dir = run_dir / variant / scenario_path.stem

                    future = executor.submit(
                        run_with_logging,
                        scenario_path,
                        scenario_output_dir,
                        scenario_index + idx,
                    )
                    futures[future] = scenario_path.stem
                    # Rate limit handling: delay between submissions
                    time.sleep(float(os.getenv("BENCHMARK_DELAY", "2.0")))

                for future in as_completed(futures):
                    scenario_id = futures[future]
                    try:
                        _, scenario_result = future.result()
                        results["scenarios"][variant][scenario_id] = scenario_result
                    except Exception as e:
                        print(f"  ❌ {scenario_id}: exception raised - {e}")
                        results["scenarios"][variant][scenario_id] = {
                            "scenario": scenario_id,
                            "error": str(e),
                        }

            if pbar:
                pbar.close()
            scenario_index += len(scenarios)
        else:
            # Sequential execution with tqdm
            for idx, scenario_path in enumerate(scenario_iter):
                scenario_id = scenario_path.stem
                scenario_output_dir = run_dir / variant / scenario_id
                start_time = time.time()

                logger.log_scenario_start(scenario_id, scenario_index + idx)

                scenario_result = run_single_benchmark(
                    scenario_path=scenario_path,
                    output_dir=scenario_output_dir,
                    agents=agents,
                    verbose=verbose,
                    parallel=parallel,
                    max_workers=max_workers,
                    multi_judge=multi_judge,
                    llm_config=llm_config,
                    judge_env=judge_env,
                    logger=logger,
                    scenario_id=scenario_id,
                )

                elapsed = time.time() - start_time
                success_count = sum(1 for r in scenario_result.get("agents", {}).values() if r.get("success"))
                logger.log_scenario_complete(scenario_id, elapsed, success_count)

                results["scenarios"][variant][scenario_id] = scenario_result

                # Rate limit handling: delay between scenarios
                time.sleep(float(os.getenv("BENCHMARK_DELAY", "2.0")))

            scenario_index += len(scenarios)

    summary_path = run_dir / "benchmark_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    logger.log_final_summary(results)

    return results


def generate_summary_report(results: dict, output_path: Path) -> None:
    """Generate a summary report of the full results."""
    lines = [
        "# ISD Agent Benchmark results summary",
        "",
        f"Run time: {results['timestamp']}",
        "",
        "## Configuration",
        f"- Variant: {', '.join(results['config']['variants'])}",
        f"- Agent: {', '.join(results['config']['agents'])}",
        "",
        "## Results summary",
        "",
    ]

    total_scenarios = 0
    agent_stats = {}

    for variant, scenarios in results.get("scenarios", {}).items():
        lines.append(f"### {variant.upper()}")
        lines.append("")

        for scenario_id, scenario_result in scenarios.items():
            total_scenarios += 1
            lines.append(f"#### {scenario_id}")
            lines.append("")

            for agent_id, agent_result in scenario_result.get("agents", {}).items():
                if agent_id not in agent_stats:
                    agent_stats[agent_id] = {"success": 0, "failed": 0}

                if agent_result.get("success"):
                    agent_stats[agent_id]["success"] += 1
                    status = "success"
                else:
                    agent_stats[agent_id]["failed"] += 1
                    status = f"failed: {agent_result.get('error', 'Unknown')[:50]}"

                lines.append(f"- {agent_id}: {status}")

            if "evaluation" in scenario_result:
                eval_data = scenario_result["evaluation"]
                lines.append("")
                lines.append("**Evaluation scores:**")

                # Extract scores from comparison rankings (in ranking order)
                comparison = eval_data.get("comparison", {})
                rankings = comparison.get("rankings", [])

                # Ensure ranking order
                rankings.sort(key=lambda x: x.get("rank", 999))

                for rank_info in rankings:
                    agent_id = rank_info.get("agent_id", "unknown")
                    total = rank_info.get("total_score", 0)
                    process = rank_info.get("process_score")

                    score_str = f"{total:.1f}/100"
                    if process is not None:
                         score_str += f" (process: {process:.1f})"

                    lines.append(f"- {agent_id}: {score_str}")

                # Fallback when rankings are missing (e.g. due to errors)
                if not rankings and "agents" in eval_data:
                     # Legacy logic (fallback)
                     for agent_score in eval_data.get("agents", []):
                        agent_id = agent_score.get("agent_id", "unknown")
                        total = agent_score.get("total", 0)
                        lines.append(f"- {agent_id}: {total:.1f}/100")

            lines.append("")

    lines.append("## Agent statistics")
    lines.append("")
    lines.append("| Agent | Success | Failed | Success rate |")
    lines.append("|-------|---------|--------|--------------|")

    for agent_id, stats in agent_stats.items():
        total = stats["success"] + stats["failed"]
        rate = (stats["success"] / total * 100) if total > 0 else 0
        lines.append(f"| {agent_id} | {stats['success']} | {stats['failed']} | {rate:.1f}% |")

    lines.append("")
    lines.append("---")
    lines.append(f"Total scenarios: {total_scenarios}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the ISD Agent Benchmark"
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install agent packages",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check agent installation status",
    )
    parser.add_argument(
        "--variant",
        "-t",
        type=str,
        default=None,
        help="Variant to run (comma-separated, e.g. idld_aligned,context_variant). Cannot be used with --dataset",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default=None,
        help="Dataset directory under scenarios/ (e.g. train, test, test_90). Cannot be used with --variant",
    )
    parser.add_argument(
        "--agents",
        "-a",
        type=str,
        default=None,
        help="Agents to run (comma-separated)",
    )
    parser.add_argument(
        "--scenario",
        "-s",
        type=str,
        default=None,
        help="Path to a specific scenario file",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    # Parallel execution options (default: maximum parallelism)
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable agent parallel execution (default: parallel)",
    )
    parser.add_argument(
        "--max-workers",
        "-w",
        type=int,
        default=6,
        help="Number of concurrent agents (default: 6, all agents at once)",
    )
    parser.add_argument(
        "--no-scenario-parallel",
        action="store_true",
        help="Disable scenario-level parallel execution (default: parallel)",
    )
    parser.add_argument(
        "--scenario-max-workers",
        type=int,
        default=8,
        help="Concurrent scenario execution count (default: 8)",
    )
    parser.add_argument(
        "--multi-judge",
        action="store_true",
        default=True,
        help="Use multi-judge evaluation (default: True)",
    )
    parser.add_argument(
        "--single-judge",
        action="store_true",
        help="Use single-judge evaluation (disable multi-judge)",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Optional label inserted into the run directory name "
             "(e.g. --run-tag r1 -> <dataset>_benchmark_<model>_r1_<timestamp>); "
             "use to distinguish repeated runs of the same model",
    )
    parser.add_argument(
        "--rate-limit",
        "-r",
        type=str,
        choices=["conservative", "moderate", "aggressive", "turbo"],
        default="conservative",
        help="Rate limit mode: conservative (2x2=4), moderate (3x4=12), aggressive (6x8=48), turbo (6x16=96). Default: conservative",
    )
    parser.add_argument(
        "--agent-model-provider",
        type=str,
        default=None,
        help="Agent model provider preset (openrouter, upstage, local-lmstudio, etc.)",
    )
    parser.add_argument(
        "--agent-model-api-spec",
        type=str,
        choices=["openai_compatible", "openai", "anthropic"],
        default=None,
        help="Agent model API spec",
    )
    parser.add_argument(
        "--agent-model-base-url",
        type=str,
        default=None,
        help="Agent model OpenAI-compatible base URL",
    )
    parser.add_argument(
        "--agent-model-name",
        type=str,
        default=None,
        help="Agent model name",
    )
    parser.add_argument(
        "--agent-model-api-key",
        type=str,
        default=None,
        help="Agent model API key value",
    )
    parser.add_argument(
        "--agent-model-api-key-env",
        type=str,
        default=None,
        help="Environment variable containing the agent model API key",
    )
    parser.add_argument(
        "--agent-model-api-key-envs",
        type=str,
        default=None,
        help="Comma-separated agent model API key env vars for round-robin credentials",
    )
    parser.add_argument(
        "--judge-model-provider",
        type=str,
        default=None,
        help="Default judge model provider preset/label",
    )
    parser.add_argument(
        "--judge-model-providers",
        type=str,
        default=None,
        help="Comma-separated provider labels, aligned with --judge-model-names",
    )
    parser.add_argument(
        "--judge-model-names",
        type=str,
        default=None,
        help="Comma-separated judge model names",
    )
    parser.add_argument(
        "--judge-model-base-url",
        type=str,
        default=None,
        help="Default judge OpenAI-compatible base URL",
    )
    parser.add_argument(
        "--judge-model-base-urls",
        type=str,
        default=None,
        help="Comma-separated judge base URLs, aligned with --judge-model-names",
    )
    parser.add_argument(
        "--judge-model-api-key",
        type=str,
        default=None,
        help="Default judge API key value",
    )
    parser.add_argument(
        "--judge-model-api-keys",
        type=str,
        default=None,
        help="Comma-separated judge API key groups, with '|' for multiple keys per judge model",
    )
    parser.add_argument(
        "--judge-model-api-key-env",
        type=str,
        default=None,
        help="Environment variable containing the default judge API key",
    )
    parser.add_argument(
        "--judge-model-api-key-envs",
        type=str,
        default=None,
        help="Comma-separated judge API key env groups, with '|' for multiple env vars per judge model",
    )
    parser.add_argument(
        "--judge-model-credential-strategies",
        type=str,
        default=None,
        help="Comma-separated judge credential strategies, aligned with --judge-model-names",
    )

    args = parser.parse_args()
    llm_config = llm_config_from_env(
        provider=args.agent_model_provider,
        api_spec=args.agent_model_api_spec,
        base_url=args.agent_model_base_url,
        model=args.agent_model_name,
        api_key=args.agent_model_api_key,
        api_key_env=args.agent_model_api_key_env,
        api_key_envs=tuple(
            part.strip()
            for part in ((args.agent_model_api_key_envs or "").split(","))
            if part.strip()
        ),
    )
    judge_env = _build_judge_env(
        provider=args.judge_model_provider,
        providers=args.judge_model_providers,
        models=args.judge_model_names,
        base_url=args.judge_model_base_url,
        base_urls=args.judge_model_base_urls,
        api_key=args.judge_model_api_key,
        api_keys=args.judge_model_api_keys,
        api_key_env=args.judge_model_api_key_env,
        api_key_envs=args.judge_model_api_key_envs,
        credential_strategies=args.judge_model_credential_strategies,
    )

    # Adjust settings based on the rate limit mode
    rate_limit_configs = {
        "conservative": {"max_workers": 2, "scenario_max_workers": 2, "delay": 2.0},
        "moderate": {"max_workers": 3, "scenario_max_workers": 4, "delay": 0.5},
        "aggressive": {"max_workers": 7, "scenario_max_workers": 8, "delay": 0.1},
        "turbo": {"max_workers": 7, "scenario_max_workers": 15, "delay": 0.0},
    }
    rate_config = rate_limit_configs[args.rate_limit]

    # Use rate_limit mode values unless explicitly overridden
    if args.max_workers == 6:  # default value
        args.max_workers = rate_config["max_workers"]
    if args.scenario_max_workers == 8:  # default value
        args.scenario_max_workers = rate_config["scenario_max_workers"]

    os.environ["BENCHMARK_DELAY"] = str(rate_config["delay"])

    if args.install:
        success = install_agents()
        sys.exit(0 if success else 1)

    if args.check:
        print("\nAgent installation status:")
        print("-" * 40)
        status = check_agents_installed()
        for agent, installed in status.items():
            icon = "✓" if installed else "✗"
            print(f"  {icon} {agent}")

        all_installed = all(status.values())
        if not all_installed:
            print("\nSome agents are not installed.")
            print("Install them with the --install option.")
        sys.exit(0 if all_installed else 1)

    status = check_agents_installed()
    if not all(status.values()):
        missing = [agent for agent, installed in status.items() if not installed]
        print(f"\n⚠️  Failed to import agent modules: {', '.join(missing)}")
        print("Check sys.path or dependencies.")
        sys.exit(1)

    # Single-scenario run
    if args.scenario:
        scenario_path = Path(args.scenario)
        if not scenario_path.exists():
            print(f"Error: scenario file not found: {scenario_path}")
            sys.exit(1)

        output_dir = RESULTS_DIR / f"single_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        agents = args.agents.split(",") if args.agents else None

        # Determine multi-judge setting
        use_multi_judge = args.multi_judge and not args.single_judge

        # Lightweight logger so a single run also shows the step checklist + eval %
        total_agents = len(agents) if agents else 6
        single_logger = BenchmarkProgressLogger(total_scenarios=1, total_agents=total_agents)

        result = run_single_benchmark(
            scenario_path=scenario_path,
            output_dir=output_dir,
            agents=agents,
            verbose=args.verbose,
            parallel=not args.no_parallel,
            max_workers=args.max_workers,
            multi_judge=use_multi_judge,
            llm_config=llm_config,
            judge_env=judge_env,
            logger=single_logger,
            scenario_id=scenario_path.stem,
        )

        print(f"\nResults saved: {output_dir}")
        sys.exit(0)

    # Prevent using --dataset and --variant together
    if args.dataset and args.variant:
        print("Error: --dataset and --variant cannot be used together.")
        print("  --dataset: named scenarios/<dataset> directory mode (e.g. train, test, test_90)")
        print("  --variant: existing variant mode (idld_aligned, context_variant)")
        sys.exit(1)

    # Run the full benchmark
    variants = args.variant.split(",") if args.variant else None
    agents = args.agents.split(",") if args.agents else None

    # Determine multi-judge setting
    use_multi_judge = args.multi_judge and not args.single_judge

    results = run_full_benchmark(
        variants=variants,
        agents=agents,
        verbose=args.verbose,
        parallel=not args.no_parallel,
        max_workers=args.max_workers,
        scenario_parallel=not args.no_scenario_parallel,
        scenario_max_workers=args.scenario_max_workers,
        dataset=args.dataset,
        multi_judge=use_multi_judge,
        llm_config=llm_config,
        judge_env=judge_env,
        run_tag=args.run_tag,
    )

    timestamp = results["timestamp"]
    output_dir = results.get("output_dir", RESULTS_DIR / f"benchmark_{timestamp}")
    summary_path = Path(output_dir) / "SUMMARY.md"
    generate_summary_report(results, summary_path)
    print(f"Summary report: {summary_path}")


if __name__ == "__main__":
    main()
