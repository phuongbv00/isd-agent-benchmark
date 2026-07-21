"""평가 메트릭 모듈

openai 의존 모듈(ADDIERubricEvaluator, TrajectoryEvaluator, CompositeEvaluator)은
lazy import로 처리하여 openai 미설치 환경에서도 ContextWeightAdjuster 사용 가능.
"""

# openai 미의존 모듈은 즉시 import
from isd_evaluator.metrics.alignment import AlignmentEvaluator, AlignmentScore
from isd_evaluator.metrics.context_weights import ContextWeightAdjuster
from isd_evaluator.models import (
    ADDIEScore,
    CompositeScore,
    TrajectoryScore,
)

__all__ = [
    # Evaluators
    "ADDIERubricEvaluator",
    "TrajectoryEvaluator",
    "CompositeEvaluator",
    "MultiJudgeEvaluator",
    "ContextWeightAdjuster",
    "AlignmentEvaluator",
    "AlignmentScore",
    # Score models
    "ADDIEScore",
    "CompositeScore",
    "TrajectoryScore",
]


def __getattr__(name: str):
    """Lazy import for openai-dependent evaluators."""
    if name == "ADDIERubricEvaluator":
        from isd_evaluator.metrics.addie_rubric import ADDIERubricEvaluator
        return ADDIERubricEvaluator
    elif name == "TrajectoryEvaluator":
        from isd_evaluator.metrics.trajectory import TrajectoryEvaluator
        return TrajectoryEvaluator
    elif name == "CompositeEvaluator":
        from isd_evaluator.metrics.composite import CompositeEvaluator
        return CompositeEvaluator
    elif name == "MultiJudgeEvaluator":
        from isd_evaluator.metrics.multi_judge import MultiJudgeEvaluator
        return MultiJudgeEvaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
