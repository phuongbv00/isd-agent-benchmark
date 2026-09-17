"""
Multi-Judge Evaluator

Uses 2 different LLMs to evaluate outputs and aggregates scores.
All evaluations use "nothink" mode (no reasoning/chain-of-thought).

Judges (경량 모델 2개):
- openai/gpt-4o-mini (OpenRouter)
- google/gemini-2.5-flash-lite (OpenRouter)
"""

import os
import statistics
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

from openai import OpenAI

from isd_evaluator.models import ADDIEScore, TrajectoryScore, CompositeScore
from isd_evaluator.metrics.addie_rubric import ADDIERubricEvaluator
from isd_evaluator.metrics.trajectory import TrajectoryEvaluator


@dataclass
class JudgeConfig:
    """Configuration for a single judge"""
    provider: str
    model: str
    api_key_env: Optional[str] = None
    api_key_envs: tuple[str, ...] = ()
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    api_keys: tuple[str, ...] = ()
    credential_strategy: str = "first"
    _credential_index: int = field(default=0, init=False, repr=False)
    _credential_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    @property
    def all_api_key_envs(self) -> tuple[str, ...]:
        envs: list[str] = []
        if self.api_key_env:
            envs.append(self.api_key_env)
        envs.extend(self.api_key_envs)
        return tuple(dict.fromkeys(envs))

    @property
    def all_api_keys(self) -> tuple[str, ...]:
        keys: list[str] = []
        if self.api_key:
            keys.append(self.api_key)
        keys.extend(self.api_keys)
        return tuple(key for key in keys if key)

    def resolve_api_key(self) -> Optional[str]:
        keys = list(self.all_api_keys)
        keys.extend(os.getenv(env) for env in self.all_api_key_envs)
        keys = [key for key in keys if key]
        if keys:
            if self.credential_strategy == "round_robin":
                with self._credential_lock:
                    key = keys[self._credential_index % len(keys)]
                    self._credential_index += 1
                    return key
            return keys[0]
        if self.base_url:
            host = urlparse(self.base_url).hostname or ""
            if host in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}:
                return "not-needed"
        return None

    def get_client(self) -> OpenAI:
        """Create OpenAI-compatible client for this judge"""
        api_key = self.resolve_api_key()
        if not api_key:
            raise ValueError(f"API key not found: {self.api_key_env}")
        if self.base_url:
            return OpenAI(api_key=api_key, base_url=self.base_url)
        return OpenAI(api_key=api_key)


# OpenRouter base URL for non-Upstage models
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Default judge configurations (2 models, all nothink)
# 경량 모델 2개: GPT-4o-mini + Gemini-2.5-Flash-Lite
DEFAULT_JUDGES: List[JudgeConfig] = [
    JudgeConfig(
        provider="openai",
        model="openai/gpt-4o-mini",  # OpenRouter model path
        api_key_env="OPENROUTER_API_KEY",
        base_url=OPENROUTER_BASE_URL,
    ),
    JudgeConfig(
        provider="google",
        model="google/gemini-2.5-flash-lite",  # OpenRouter model path
        api_key_env="OPENROUTER_API_KEY",
        base_url=OPENROUTER_BASE_URL,
    ),
]


def _split_env(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _split_grouped_env(value: Optional[str]) -> List[tuple[str, ...]]:
    """Split comma-separated judge groups, with pipe-separated values per judge."""
    if not value:
        return []
    groups: List[tuple[str, ...]] = []
    for group in value.split(","):
        values = tuple(part.strip() for part in group.split("|") if part.strip())
        groups.append(values)
    return groups


def _default_base_url_for_profile(profile: str) -> Optional[str]:
    profile = profile.lower()
    if profile in {"openrouter", "openai", "google", "deepseek", "anthropic"}:
        return OPENROUTER_BASE_URL
    if profile == "upstage":
        return "https://api.upstage.ai/v1/solar"
    if profile in {"local-ollama", "ollama"}:
        return "http://localhost:11434/v1"
    if profile in {"local-lmstudio", "lmstudio"}:
        return "http://localhost:1234/v1"
    if profile in {"local-vllm", "vllm"}:
        return "http://localhost:8000/v1"
    return None


def _default_api_key_env_for_profile(profile: str) -> Optional[str]:
    profile = profile.lower()
    if profile == "upstage":
        return "UPSTAGE_API_KEY"
    if profile in {"local-ollama", "ollama", "local-lmstudio", "lmstudio", "local-vllm", "vllm"}:
        return None
    return "OPENROUTER_API_KEY"


def _provider_for_model(model: str, fallback: str) -> str:
    if "/" in model:
        return model.split("/", 1)[0]
    return fallback


def load_judges_from_env() -> Optional[List[JudgeConfig]]:
    """Load judge model configuration from JUDGE_MODEL_* env vars.

    Returns None when no explicit judge override is present, so callers can use
    DEFAULT_JUDGES unchanged.
    """
    models = _split_env(os.getenv("JUDGE_MODEL_NAMES") or os.getenv("JUDGE_MODEL_NAME"))
    if not models:
        return None

    profile = os.getenv("JUDGE_MODEL_PROVIDER", "openrouter")
    providers = _split_env(os.getenv("JUDGE_MODEL_PROVIDERS") or os.getenv("JUDGE_MODEL_PROVIDER"))
    api_key_env_groups = _split_grouped_env(
        os.getenv("JUDGE_MODEL_API_KEY_ENVS") or os.getenv("JUDGE_MODEL_API_KEY_ENV")
    )
    api_key_groups = _split_grouped_env(
        os.getenv("JUDGE_MODEL_API_KEYS") or os.getenv("JUDGE_MODEL_API_KEY")
    )
    base_urls = _split_env(os.getenv("JUDGE_MODEL_BASE_URLS") or os.getenv("JUDGE_MODEL_BASE_URL"))
    credential_strategies = _split_env(os.getenv("JUDGE_MODEL_CREDENTIAL_STRATEGIES"))

    default_base_url = _default_base_url_for_profile(profile)
    default_api_key_env = _default_api_key_env_for_profile(profile)

    judges = []
    for idx, model in enumerate(models):
        provider = providers[idx] if idx < len(providers) else _provider_for_model(model, profile)
        api_key_env_group = api_key_env_groups[idx] if idx < len(api_key_env_groups) else ()
        api_key_group = api_key_groups[idx] if idx < len(api_key_groups) else ()
        credential_strategy = (
            credential_strategies[idx]
            if idx < len(credential_strategies)
            else ("round_robin" if len(api_key_env_group) + len(api_key_group) > 1 else "first")
        )
        judges.append(JudgeConfig(
            provider=provider,
            model=model,
            api_key_env=api_key_env_group[0] if api_key_env_group else default_api_key_env,
            api_key_envs=api_key_env_group[1:] if api_key_env_group else (),
            api_key=api_key_group[0] if api_key_group else None,
            api_keys=api_key_group[1:] if api_key_group else (),
            credential_strategy=credential_strategy,
            base_url=base_urls[idx] if idx < len(base_urls) else default_base_url,
        ))

    return judges


@dataclass
class JudgeResult:
    """Result from a single judge"""
    provider: str
    model: str
    addie_score: Optional[ADDIEScore] = None
    trajectory_score: Optional[TrajectoryScore] = None
    error: Optional[str] = None

    @property
    def normalized_score(self) -> Optional[float]:
        """Get normalized ADDIE score (0-100)"""
        if self.addie_score:
            return self.addie_score.normalized_score
        return None

    @property
    def trajectory_total(self) -> Optional[float]:
        """Get total trajectory score (0-100)"""
        if self.trajectory_score:
            return self.trajectory_score.total
        return None


@dataclass
class MultiJudgeResult:
    """Aggregated result from multiple judges"""
    judge_results: List[JudgeResult]
    median_addie_score: float
    median_trajectory_score: Optional[float]
    agreement_stats: Dict[str, Any]

    @property
    def total_score(self) -> float:
        """Weighted total: ADDIE 70% + Trajectory 30%"""
        if self.median_trajectory_score is not None:
            return self.median_addie_score * 0.7 + self.median_trajectory_score * 0.3
        return self.median_addie_score

    @property
    def mean_addie_score(self) -> float:
        """Mean ADDIE score across judges"""
        return self.agreement_stats.get("addie_mean", self.median_addie_score)

    def representative_judge(self) -> Optional[JudgeResult]:
        """Judge result closest to the median ADDIE score."""
        valid_results = [
            r for r in self.judge_results
            if r.normalized_score is not None
        ]
        if not valid_results:
            return None
        return min(
            valid_results,
            key=lambda r: abs((r.normalized_score or 0) - self.median_addie_score),
        )

    @property
    def reliability_score(self) -> float:
        """
        Reliability score (0-1) based on inter-judge agreement.
        Higher = more reliable (lower variance)
        """
        stdev = self.agreement_stats.get("addie_stdev", 0)
        # Normalize: stdev of 0 = 1.0 reliability, stdev of 20+ = 0.0
        return max(0.0, 1.0 - (stdev / 20.0))

    def to_dict(self) -> dict:
        """Convert to dictionary with full statistics"""
        representative = self.representative_judge()
        details = representative.addie_score.to_dict() if representative and representative.addie_score else {}
        phase_scores = {
            phase: data.get("percentage", 0)
            for phase, data in details.get("phases", {}).items()
        }
        trajectory_details = (
            representative.trajectory_score.to_dict()
            if representative and representative.trajectory_score
            else None
        )

        return {
            # Aggregated scores
            "scores": {
                "total": round(self.total_score, 2),
                "addie_median": round(self.median_addie_score, 2),
                "addie_mean": round(self.mean_addie_score, 2),
                "trajectory_median": round(self.median_trajectory_score, 2) if self.median_trajectory_score else None,
            },
            # Reliability & Agreement
            "reliability": {
                "score": round(self.reliability_score, 3),
                "interpretation": self._interpret_reliability(),
                "stdev": round(self.agreement_stats.get("addie_stdev", 0), 2),
                "range": round(self.agreement_stats.get("addie_range", 0), 2),
                "min": round(self.agreement_stats.get("addie_min", 0), 2),
                "max": round(self.agreement_stats.get("addie_max", 0), 2),
                "high_disagreement": self.agreement_stats.get("addie_high_disagreement", False),
            },
            # Individual judge scores
            "judges": [
                {
                    "provider": r.provider,
                    "model": r.model,
                    "addie_score": round(r.normalized_score, 2) if r.normalized_score else None,
                    "trajectory_score": round(r.trajectory_total, 2) if r.trajectory_total else None,
                    "deviation_from_median": round(r.normalized_score - self.median_addie_score, 2) if r.normalized_score else None,
                    "error": r.error,
                }
                for r in self.judge_results
            ],
            # Representative judge details for phase/item reporting
            "phase_scores": phase_scores,
            "details": details,
            "trajectory_details": trajectory_details,
            # Meta
            "num_judges": self.agreement_stats.get("num_judges", 0),
            "num_valid": self.agreement_stats.get("num_valid_addie", 0),
        }

    def _interpret_reliability(self) -> str:
        """Interpret reliability score"""
        r = self.reliability_score
        if r >= 0.9:
            return "Excellent (very high agreement)"
        elif r >= 0.75:
            return "Good (high agreement)"
        elif r >= 0.5:
            return "Moderate (acceptable agreement)"
        elif r >= 0.25:
            return "Low (consider review)"
        else:
            return "Poor (requires human review)"

    def print_summary(self) -> str:
        """Generate printable summary"""
        lines = [
            "=" * 70,
            "MULTI-JUDGE EVALUATION RESULTS",
            "=" * 70,
            "",
            f"📊 AGGREGATED SCORES",
            f"   Total Score:     {self.total_score:.1f}/100",
            f"   ADDIE (median):  {self.median_addie_score:.1f}/100",
            f"   ADDIE (mean):    {self.mean_addie_score:.1f}/100",
        ]

        if self.median_trajectory_score:
            lines.append(f"   Trajectory:      {self.median_trajectory_score:.1f}/100")

        lines.extend([
            "",
            f"🎯 RELIABILITY",
            f"   Score:           {self.reliability_score:.2f} ({self._interpret_reliability()})",
            f"   Std Dev:         {self.agreement_stats.get('addie_stdev', 0):.2f}",
            f"   Range:           {self.agreement_stats.get('addie_min', 0):.1f} - {self.agreement_stats.get('addie_max', 0):.1f}",
        ])

        if self.agreement_stats.get("addie_high_disagreement"):
            lines.append("   ⚠️  HIGH DISAGREEMENT - Consider human review")

        lines.extend([
            "",
            f"👨‍⚖️ INDIVIDUAL JUDGE SCORES",
            "-" * 50,
        ])

        for jr in sorted(self.judge_results, key=lambda x: x.normalized_score or 0, reverse=True):
            if jr.normalized_score is not None:
                deviation = jr.normalized_score - self.median_addie_score
                dev_str = f"+{deviation:.1f}" if deviation >= 0 else f"{deviation:.1f}"
                lines.append(f"   {jr.provider:12} {jr.model:30} {jr.normalized_score:5.1f}  ({dev_str})")
            else:
                lines.append(f"   {jr.provider:12} {jr.model:30} ERROR: {jr.error}")

        lines.extend(["", "=" * 70])
        return "\n".join(lines)


class MultiJudgeEvaluator:
    """
    Multi-Judge Evaluator using 2 lightweight LLMs.

    Aggregates scores using median for robustness against outliers.
    All evaluations use nothink mode:
    - temperature=0
    - reasoning.enabled=False (via OpenRouter extra_body)
    - No chain-of-thought prompting
    """

    def __init__(
        self,
        judges: Optional[List[JudgeConfig]] = None,
        parallel: bool = True,
        max_workers: int = 10,
        include_benchmarks: bool = True,
    ):
        """
        Args:
            judges: List of judge configurations (default: 5 models)
            parallel: Run judges in parallel
            max_workers: Maximum parallel workers
            include_benchmarks: Include benchmark examples in prompts
        """
        self.judges = judges or load_judges_from_env() or DEFAULT_JUDGES
        self.parallel = parallel
        self.max_workers = max_workers
        self.include_benchmarks = include_benchmarks

        # Validate API keys
        self._validate_api_keys()

    def _validate_api_keys(self):
        """Check if required API keys are available"""
        missing = []
        for judge in self.judges:
            if not judge.resolve_api_key():
                missing.append(f"{judge.provider}: {judge.api_key_env or 'JUDGE_MODEL_API_KEY'}")

        if missing:
            print(f"[MultiJudgeEvaluator] Warning: Missing API keys: {missing}")

    def _evaluate_with_judge(
        self,
        judge: JudgeConfig,
        addie_output: dict,
        scenario: Optional[dict] = None,
        trajectory: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ) -> JudgeResult:
        """Run evaluation with a single judge"""
        try:
            # Create ADDIE evaluator for this judge
            addie_evaluator = ADDIERubricEvaluator(
                model=judge.model,
                api_key=judge.resolve_api_key(),
                provider=judge.provider,
                base_url=judge.base_url,
                api_key_env=judge.api_key_env,
                include_benchmarks=self.include_benchmarks,
                temperature=0.0,  # nothink mode
            )

            # Evaluate ADDIE output
            addie_score = addie_evaluator.evaluate(addie_output, scenario)

            # Evaluate trajectory if provided
            trajectory_score = None
            if trajectory:
                traj_evaluator = TrajectoryEvaluator(
                    model=judge.model,
                    api_key=judge.resolve_api_key(),
                    provider=judge.provider,
                    base_url=judge.base_url,
                    api_key_env=judge.api_key_env,
                )
                trajectory_score = traj_evaluator.evaluate(trajectory, metadata)

            return JudgeResult(
                provider=judge.provider,
                model=judge.model,
                addie_score=addie_score,
                trajectory_score=trajectory_score,
            )

        except Exception as e:
            return JudgeResult(
                provider=judge.provider,
                model=judge.model,
                error=str(e),
            )

    def evaluate(
        self,
        addie_output: dict,
        scenario: Optional[dict] = None,
        trajectory: Optional[dict] = None,
        metadata: Optional[dict] = None,
        on_judge_done: Optional[Callable[[JudgeResult], None]] = None,
    ) -> MultiJudgeResult:
        """
        Evaluate using all judges and aggregate results.

        Args:
            addie_output: ADDIE output to evaluate
            scenario: Original scenario
            trajectory: Agent trajectory
            metadata: Metadata
            on_judge_done: Optional callback invoked with each JudgeResult as it
                completes (used to report per-judge-call progress).

        Returns:
            MultiJudgeResult with median scores and agreement stats
        """
        judge_results: List[JudgeResult] = []

        if self.parallel:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(
                        self._evaluate_with_judge,
                        judge,
                        addie_output,
                        scenario,
                        trajectory,
                        metadata,
                    ): judge
                    for judge in self.judges
                }

                for future in as_completed(futures):
                    judge = futures[future]
                    try:
                        result = future.result()
                        judge_results.append(result)
                        print(f"  [{judge.provider}] {judge.model}: "
                              f"ADDIE={result.normalized_score:.1f}" if result.normalized_score else f"  [{judge.provider}] Error")
                    except Exception as e:
                        result = JudgeResult(
                            provider=judge.provider,
                            model=judge.model,
                            error=str(e),
                        )
                        judge_results.append(result)
                    if on_judge_done is not None:
                        on_judge_done(result)
        else:
            # Sequential execution
            for judge in self.judges:
                result = self._evaluate_with_judge(
                    judge, addie_output, scenario, trajectory, metadata
                )
                judge_results.append(result)
                print(f"  [{judge.provider}] {judge.model}: "
                      f"ADDIE={result.normalized_score:.1f}" if result.normalized_score else f"  [{judge.provider}] Error")
                if on_judge_done is not None:
                    on_judge_done(result)

        return self._aggregate_results(judge_results)

    def _aggregate_results(self, judge_results: List[JudgeResult]) -> MultiJudgeResult:
        """Aggregate results from all judges using median"""

        # Collect valid scores
        addie_scores = [
            r.normalized_score for r in judge_results
            if r.normalized_score is not None
        ]
        trajectory_scores = [
            r.trajectory_total for r in judge_results
            if r.trajectory_total is not None
        ]

        # Calculate median
        median_addie = statistics.median(addie_scores) if addie_scores else 0.0
        median_trajectory = statistics.median(trajectory_scores) if trajectory_scores else None

        # Calculate agreement statistics
        agreement_stats = self._calculate_agreement_stats(addie_scores, trajectory_scores)

        return MultiJudgeResult(
            judge_results=judge_results,
            median_addie_score=median_addie,
            median_trajectory_score=median_trajectory,
            agreement_stats=agreement_stats,
        )

    def _calculate_agreement_stats(
        self,
        addie_scores: List[float],
        trajectory_scores: List[float],
    ) -> Dict[str, Any]:
        """Calculate inter-judge agreement statistics"""
        stats = {
            "num_judges": len(self.judges),
            "num_valid_addie": len(addie_scores),
            "num_valid_trajectory": len(trajectory_scores),
        }

        if len(addie_scores) >= 2:
            stats["addie_mean"] = statistics.mean(addie_scores)
            stats["addie_stdev"] = statistics.stdev(addie_scores)
            stats["addie_range"] = max(addie_scores) - min(addie_scores)
            stats["addie_min"] = min(addie_scores)
            stats["addie_max"] = max(addie_scores)

            # Flag high disagreement (stdev > 10 or range > 20)
            stats["addie_high_disagreement"] = (
                stats["addie_stdev"] > 10 or stats["addie_range"] > 20
            )

        if len(trajectory_scores) >= 2:
            stats["trajectory_mean"] = statistics.mean(trajectory_scores)
            stats["trajectory_stdev"] = statistics.stdev(trajectory_scores)
            stats["trajectory_range"] = max(trajectory_scores) - min(trajectory_scores)

        return stats

    def compare_agents(
        self,
        results: List[dict],
        scenario: Optional[dict] = None,
    ) -> dict:
        """
        Compare multiple agents using multi-judge evaluation.

        Args:
            results: List of agent results [{agent_id, addie_output, trajectory, metadata}, ...]
            scenario: Common scenario

        Returns:
            Comparison results with rankings and detailed statistics
        """
        evaluations = []

        # Progress markers (consumed by the benchmark runner) emit one line per
        # completed judge-call so progress can be shown as N_agents x N_judges.
        # The last field is the judge model name (not the provider).
        progress_enabled = bool(os.getenv("ISD_EVAL_PROGRESS"))
        total_units = len(results) * len(self.judges)
        progress_counter = {"done": 0}
        progress_lock = threading.Lock()

        def make_emit(agent_id: str) -> Callable[[JudgeResult], None]:
            def emit(jr: JudgeResult) -> None:
                if not progress_enabled:
                    return
                with progress_lock:
                    progress_counter["done"] += 1
                    done = progress_counter["done"]
                print(
                    f"__EVAL_PROGRESS__\t{done}\t{total_units}\t{agent_id}\t{jr.model}",
                    flush=True,
                )
            return emit

        for result in results:
            agent_id = result.get("agent_id", "unknown")
            print(f"\n[MultiJudge] Evaluating {agent_id}...")

            multi_result = self.evaluate(
                addie_output=result.get("addie_output", {}),
                scenario=scenario,
                trajectory=result.get("trajectory"),
                metadata=result.get("metadata"),
                on_judge_done=make_emit(agent_id),
            )

            evaluations.append({
                "agent_id": agent_id,
                "result": multi_result,
            })

        # Sort by total score
        evaluations.sort(key=lambda x: x["result"].total_score, reverse=True)

        # Build detailed rankings
        rankings = []
        for i, e in enumerate(evaluations):
            r = e["result"]
            result_dict = r.to_dict()
            rankings.append({
                "rank": i + 1,
                "agent_id": e["agent_id"],
                # Scores
                "total_score": round(r.total_score, 2),
                "addie_median": round(r.median_addie_score, 2),
                "addie_mean": round(r.mean_addie_score, 2),
                "trajectory_score": round(r.median_trajectory_score, 2) if r.median_trajectory_score else None,
                # Reliability
                "reliability_score": round(r.reliability_score, 3),
                "reliability": r._interpret_reliability(),
                "stdev": round(r.agreement_stats.get("addie_stdev", 0), 2),
                "score_range": f"{r.agreement_stats.get('addie_min', 0):.1f}-{r.agreement_stats.get('addie_max', 0):.1f}",
                "high_disagreement": r.agreement_stats.get("addie_high_disagreement", False),
                # Individual judges
                "judges": result_dict["judges"],
                # Representative judge details for report tables
                "phase_scores": result_dict.get("phase_scores", {}),
                "details": result_dict.get("details", {}),
                "trajectory_details": result_dict.get("trajectory_details"),
            })

        return {
            "rankings": rankings,
            "best_agent": evaluations[0]["agent_id"] if evaluations else None,
            "evaluation_method": "multi-judge",
            "judge_models": [j.model for j in self.judges],
            "aggregation": "median",
        }


def analyze_self_preference_bias(
    agent_outputs: Dict[str, dict],
    scenario: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Analyze self-preference bias in multi-judge evaluation.

    Compares scores when judge model family matches agent model family
    vs. when they differ.

    Args:
        agent_outputs: {agent_id: {addie_output, trajectory, metadata, model_family}}
        scenario: Common scenario

    Returns:
        Bias analysis results
    """
    evaluator = MultiJudgeEvaluator()

    # Cross-evaluation matrix
    cross_matrix = {}  # {agent_family: {judge_family: score}}

    for agent_id, agent_data in agent_outputs.items():
        agent_family = agent_data.get("model_family", "unknown")
        cross_matrix[agent_family] = {}

        # Evaluate with each judge
        for judge in evaluator.judges:
            result = evaluator._evaluate_with_judge(
                judge,
                agent_data.get("addie_output", {}),
                scenario,
                agent_data.get("trajectory"),
                agent_data.get("metadata"),
            )

            if result.normalized_score is not None:
                cross_matrix[agent_family][judge.provider] = result.normalized_score

    # Calculate bias
    bias_results = {}
    for agent_family, scores in cross_matrix.items():
        if agent_family in scores:
            self_score = scores[agent_family]
            other_scores = [s for f, s in scores.items() if f != agent_family]

            if other_scores:
                other_mean = statistics.mean(other_scores)
                bias = self_score - other_mean

                bias_results[agent_family] = {
                    "self_score": self_score,
                    "other_mean": other_mean,
                    "bias": bias,
                    "bias_percentage": (bias / other_mean * 100) if other_mean else 0,
                }

    return {
        "cross_matrix": cross_matrix,
        "bias_analysis": bias_results,
    }
