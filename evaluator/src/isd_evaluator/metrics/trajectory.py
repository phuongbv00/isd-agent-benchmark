"""
궤적(Trajectory) 평가 메트릭 - LLM 기반 질적 평가

Agent의 생성 과정을 LLM을 통해 질적으로 평가합니다.
규칙 기반 횟수/시도 평가가 아닌, 실제 추론 과정의 질을 평가합니다.

Level 2 (Process Evaluation):
- Single Agent: 추론의 논리성 (CoT Coherence)
- Multi Agent: 협업 기여도 (Collaboration Gain)
"""

import json
import os
import re
from pathlib import Path
from typing import Optional

# .env 파일에서 환경변수 로드 (프로젝트 루트 기준)
try:
    from dotenv import load_dotenv
    _project_root = Path(__file__).parent.parent.parent.parent.parent
    _env_path = _project_root / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass

from openai import OpenAI

# TrajectoryScore 모델은 중앙 models에서 import
from isd_evaluator.models import TrajectoryScore


# LLM 기반 궤적 평가 프롬프트 (BFCL 기반 도구 사용 평가)
TRAJECTORY_EVALUATION_PROMPT = """당신은 AI Agent의 도구 사용(Tool Use)을 평가하는 전문가입니다.
Berkeley Function-Calling Leaderboard(BFCL) 기준에 따라 Agent의 도구 사용 품질을 4가지 차원에서 평가해주세요.

⚠️ **평가 원칙**:
1. 단순 호출 횟수가 아닌, **도구 사용의 질적 수준**을 평가하세요
2. 각 항목을 **0.0 ~ 25.0 점 사이의 소수점** 단위로 평가하세요 (예: 18.5, 21.2)
3. 싱글/멀티 에이전트 구분 없이 **동일한 기준**으로 공정하게 평가하세요
4. 도구 선택, 인자 전달, 결과 활용의 전 과정을 종합적으로 평가하세요

## 점수 기준 (각 차원 0-25점)
- **23-25점**: 탁월함 - 완벽한 도구 선택과 활용, 오류 없음
- **18-22점**: 우수함 - 대부분 올바른 도구 사용, 경미한 개선점
- **13-17점**: 양호함 - 기본적인 도구 사용 능력 확보
- **8-12점**: 미흡함 - 잘못된 도구 선택 또는 비효율적 사용 다수
- **0-7점**: 부족함 - 도구 사용 능력 부재

## Agent 메타데이터
{metadata}

## 실행 궤적 (Trajectory)
{trajectory}

## 평가 항목

### 1. 도구 정확성 (Tool Correctness) - 0~25점
- 주어진 목적에 **올바른 도구**를 선택했는가?
- ADDIE 각 단계(분석/설계/개발/실행/평가)에 적합한 도구를 사용했는가?
- 사용 가능한 도구 중 **최적의 도구**를 선택했는가?
- 불필요하거나 관련 없는 도구를 호출하지 않았는가? (Relevance Detection)

### 2. 인자 정확성 (Argument Accuracy) - 0~25점
- 도구에 전달한 **파라미터 값이 정확**한가?
- 파라미터의 **타입과 형식**이 올바른가?
- 필수 파라미터를 빠뜨리지 않았는가?
- 이전 도구 결과를 다음 도구 인자로 **정확하게 전달**했는가?

### 3. 중복 회피 (Redundancy Avoidance) - 0~25점
- **불필요한 중복 호출** 없이 효율적으로 도구를 사용했는가?
- 동일한 도구를 같은 인자로 반복 호출하지 않았는가?
- 한 번의 호출로 충분한 작업을 여러 번 나눠서 호출하지 않았는가?
- 전체 도구 호출 수가 작업 복잡도 대비 적절한가?

### 4. 결과 활용도 (Result Utilization) - 0~25점
- 도구의 출력 결과를 **다음 추론에 효과적으로 활용**했는가?
- 도구 결과를 무시하거나 잘못 해석하지 않았는가?
- 도구 실패 시 **적절한 오류 복구**(재시도, 대안 도구)를 수행했는가?
- 최종 산출물에 도구 결과가 **일관성 있게 반영**되었는가?

## 출력 형식
반드시 아래 JSON 형식으로만 응답하세요. 모든 점수는 소수점 첫째 자리까지 표기하세요.

```json
{{
  "scores": {{
    "tool_correctness": <0.0-25.0>,
    "argument_accuracy": <0.0-25.0>,
    "redundancy_avoidance": <0.0-25.0>,
    "result_utilization": <0.0-25.0>
  }},
  "reasoning": {{
    "tool_correctness": "<평가 근거>",
    "argument_accuracy": "<평가 근거>",
    "redundancy_avoidance": "<평가 근거>",
    "result_utilization": "<평가 근거>"
  }},
  "overall_assessment": "<전체적인 도구 사용 평가 요약>"
}}
```
"""


# API Configuration (5 providers)
# Upstage uses UPSTAGE_API_KEY, others use OPENROUTER_API_KEY
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

PROVIDER_CONFIG = {
    "upstage": {
        "base_url": "https://api.upstage.ai/v1/solar",
        "default_model": "solar-pro3",
        "api_key_env": "UPSTAGE_API_KEY",
    },
    "openai": {
        "base_url": OPENROUTER_BASE_URL,
        "default_model": "openai/gpt-5.2",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "google": {
        "base_url": OPENROUTER_BASE_URL,
        "default_model": "google/gemini-3-pro-preview",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "deepseek": {
        "base_url": OPENROUTER_BASE_URL,
        "default_model": "deepseek/deepseek-v3.2",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "anthropic": {
        "base_url": OPENROUTER_BASE_URL,
        "default_model": "anthropic/claude-opus-4.5",
        "api_key_env": "OPENROUTER_API_KEY",
    },
}


class TrajectoryEvaluator:
    """LLM-based trajectory evaluator"""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key_env: Optional[str] = None,
    ):
        """
        Args:
            model: LLM model name
            api_key: API key
            provider: "upstage", "openai", "google", "deepseek", "anthropic"
        """
        self.provider = provider or os.getenv("TRAJ_EVAL_PROVIDER") or os.getenv("JUDGE_MODEL_PROVIDER", "upstage")

        # Get provider config
        config = PROVIDER_CONFIG.get(self.provider, PROVIDER_CONFIG["openai"])

        self.model = model or os.getenv("TRAJ_EVAL_MODEL", config["default_model"])
        base_url = base_url or os.getenv("TRAJ_EVAL_BASE_URL") or os.getenv("JUDGE_MODEL_BASE_URL") or config["base_url"]
        api_key_env = api_key_env or os.getenv("TRAJ_EVAL_API_KEY_ENV") or os.getenv("JUDGE_MODEL_API_KEY_ENV") or config["api_key_env"]
        resolved_api_key = api_key or os.getenv("TRAJ_EVAL_API_KEY") or os.getenv("JUDGE_MODEL_API_KEY") or os.getenv(api_key_env or "")

        if not resolved_api_key and base_url:
            from urllib.parse import urlparse
            host = urlparse(base_url).hostname or ""
            if host in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}:
                resolved_api_key = "not-needed"

        # Create client
        client_kwargs = {
            "api_key": resolved_api_key,
        }
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)

    def evaluate(
        self,
        trajectory: dict,
        metadata: Optional[dict] = None,
    ) -> TrajectoryScore:
        """
        Agent 궤적을 LLM으로 평가합니다.

        Args:
            trajectory: 궤적 딕셔너리 (tool_calls, reasoning_steps, agent_interactions 등)
                       또는 전체 trajectory 파일 데이터 (중첩 구조 자동 처리)
            metadata: 메타데이터 (실행 시간, 토큰, 반복 횟수 등)

        Returns:
            TrajectoryScore: 평가 점수
        """
        # 중첩 구조 처리: trajectory 키가 있으면 내부 데이터 사용
        if "trajectory" in trajectory and isinstance(trajectory["trajectory"], dict):
            inner_trajectory = trajectory["trajectory"]
            # 파일에서 메타데이터 추출 (명시적으로 전달되지 않은 경우)
            if metadata is None and "metadata" in trajectory:
                metadata = trajectory["metadata"]
            if metadata is None:
                metadata = {
                    "agent_id": trajectory.get("agent_id", "unknown"),
                    "scenario_id": trajectory.get("scenario_id", "unknown"),
                }
        else:
            inner_trajectory = trajectory

        # 궤적과 메타데이터 포맷팅
        trajectory_text = self._format_trajectory(inner_trajectory)
        metadata_text = self._format_metadata(metadata)

        # 프롬프트 구성
        prompt = TRAJECTORY_EVALUATION_PROMPT.format(
            trajectory=trajectory_text,
            metadata=metadata_text,
        )

        # LLM 호출 (nothink mode: reasoning 비활성화)
        api_params = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }
        if self.model.startswith("gpt-5") or "gpt-5" in self.model:
            api_params["max_completion_tokens"] = 2048
        else:
            api_params["max_tokens"] = 2048

        # OpenRouter nothink mode: reasoning 비활성화 (속도 향상)
        if self.provider != "upstage":
            api_params["extra_body"] = {"reasoning": {"enabled": False}}

        try:
            response = self.client.chat.completions.create(**api_params)
            content = response.choices[0].message.content
            return self._parse_scores(content)
        except Exception as e:
            # LLM 호출 실패 시 기본값 반환
            print(f"Trajectory LLM 평가 실패: {e}")
            return TrajectoryScore(
                tool_correctness=12.5,
                argument_accuracy=12.5,
                redundancy_avoidance=12.5,
                result_utilization=12.5,
            )

    def _format_trajectory(self, trajectory: dict) -> str:
        """궤적을 읽기 쉬운 형식으로 변환"""
        parts = []

        # Tool Calls
        tool_calls = trajectory.get("tool_calls", [])
        if tool_calls:
            parts.append("### Tool Calls")
            for i, tc in enumerate(tool_calls[:20], 1):  # 최대 20개
                if isinstance(tc, dict):
                    tool_name = tc.get("tool", "unknown")
                    args = tc.get("args", {})
                    result_summary = tc.get("result", "")[:200] if tc.get("result") else ""
                    parts.append(f"{i}. {tool_name}")
                    if args:
                        parts.append(f"   Args: {json.dumps(args, ensure_ascii=False)[:200]}")
                    if result_summary:
                        parts.append(f"   Result: {result_summary}...")

        # Reasoning Steps
        reasoning_steps = trajectory.get("reasoning_steps", [])
        if reasoning_steps:
            parts.append("\n### Reasoning Steps")
            for i, step in enumerate(reasoning_steps[:15], 1):  # 최대 15개
                if isinstance(step, str):
                    parts.append(f"{i}. {step[:300]}...")
                elif isinstance(step, dict):
                    parts.append(f"{i}. {json.dumps(step, ensure_ascii=False)[:300]}...")

        # Agent Interactions (Multi-Agent)
        agent_interactions = trajectory.get("agent_interactions", [])
        if agent_interactions:
            parts.append("\n### Agent Interactions")
            for i, interaction in enumerate(agent_interactions[:20], 1):  # 최대 20개
                if isinstance(interaction, dict):
                    agent = interaction.get("agent", "unknown")
                    action = interaction.get("action", "")
                    iteration = interaction.get("iteration", 0)
                    output_data = interaction.get("output_data", {})
                    score = output_data.get("score", "") if isinstance(output_data, dict) else ""
                    parts.append(f"{i}. [{iteration}회차] {agent}: {action}")
                    if score:
                        parts.append(f"   Score: {score}")

        return "\n".join(parts) if parts else "궤적 정보 없음"

    def _format_metadata(self, metadata: Optional[dict]) -> str:
        """메타데이터 포맷팅"""
        if not metadata:
            return "메타데이터 없음"

        parts = []
        if "iterations" in metadata:
            parts.append(f"- 반복 횟수: {metadata['iterations']}")
        if "execution_time_seconds" in metadata:
            parts.append(f"- 실행 시간: {metadata['execution_time_seconds']:.1f}초")
        if "total_tokens" in metadata:
            parts.append(f"- 총 토큰: {metadata['total_tokens']}")
        if "agent_id" in metadata:
            parts.append(f"- Agent ID: {metadata['agent_id']}")

        return "\n".join(parts) if parts else "메타데이터 없음"

    def _parse_scores(self, content: str) -> TrajectoryScore:
        """LLM 응답에서 점수 파싱"""
        # 기본값
        scores = {
            "tool_correctness": 12.5,
            "argument_accuracy": 12.5,
            "redundancy_avoidance": 12.5,
            "result_utilization": 12.5,
        }

        try:
            # JSON 블록 추출
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_match = re.search(r'\{[\s\S]*\}', content)
                json_str = json_match.group(0) if json_match else None

            if json_str:
                data = json.loads(json_str)
                score_data = data.get("scores", {})

                for key in scores.keys():
                    if key in score_data:
                        val = float(score_data[key])
                        scores[key] = round(min(25.0, max(0.0, val)), 1)

        except (json.JSONDecodeError, KeyError, TypeError, AttributeError, ValueError):
            pass  # 파싱 실패 시 기본값 반환

        return TrajectoryScore(**scores)
