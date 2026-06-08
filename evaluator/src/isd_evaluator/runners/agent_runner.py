"""
Agent Runner

CLI를 통해 교수설계 Agent를 실행합니다.
"""

import json
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional


AGENT_COMMANDS = {
    "eduplanner": "eduplanner",
    "baseline": "baseline",
    "react-isd": "react-isd",
    "addie-agent": "addie-agent",
    "dick-carey-agent": "dick-carey-agent",
    "rpisd-agent": "rpisd-agent",
}


class AgentRunner:
    """Agent 실행기"""

    def __init__(self, timeout: int = 300):
        """
        Args:
            timeout: 실행 타임아웃 (초)
        """
        self.timeout = timeout

    def run_agent(
        self,
        agent_id: str,
        scenario_path: Path,
        output_dir: Optional[Path] = None,
    ) -> dict:
        """
        Agent를 CLI로 실행합니다.

        Args:
            agent_id: Agent 식별자 (eduplanner, baseline, react-isd, addie-agent,
                dick-carey-agent, rpisd-agent)
            scenario_path: 시나리오 JSON 파일 경로
            output_dir: 출력 디렉토리 (선택)

        Returns:
            실행 결과 {addie_output, trajectory, metadata, success, error}
        """
        if agent_id not in AGENT_COMMANDS:
            return {
                "success": False,
                "error": f"Unknown agent: {agent_id}. Available: {list(AGENT_COMMANDS.keys())}",
            }

        command = AGENT_COMMANDS[agent_id]

        # 출력 파일 경로 설정
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{agent_id}_output.json"
            trajectory_path = output_dir / f"{agent_id}_trajectory.json"
        else:
            # 임시 디렉토리 사용
            temp_dir = Path(tempfile.mkdtemp())
            output_path = temp_dir / "output.json"
            trajectory_path = temp_dir / "trajectory.json"

        # CLI 명령 구성
        cmd = [
            command,
            "run",
            "--input", str(scenario_path),
            "--output", str(output_path),
            "--trajectory", str(trajectory_path),
        ]

        try:
            # 명령 실행
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            if result.returncode != 0:
                return {
                    "success": False,
                    "error": f"Agent failed: {result.stderr}",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }

            # 결과 로드
            addie_output = {}
            trajectory = {}
            metadata = {}

            if output_path.exists():
                with open(output_path, "r", encoding="utf-8") as f:
                    addie_output = json.load(f)

            if trajectory_path.exists():
                with open(trajectory_path, "r", encoding="utf-8") as f:
                    traj_data = json.load(f)
                    trajectory = traj_data.get("trajectory", {})
                    metadata = traj_data.get("metadata", {})

            return {
                "success": True,
                "agent_id": agent_id,
                "addie_output": addie_output,
                "trajectory": trajectory,
                "metadata": metadata,
                "output_path": str(output_path),
                "trajectory_path": str(trajectory_path),
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Agent timed out after {self.timeout} seconds",
            }
        except FileNotFoundError:
            return {
                "success": False,
                "error": f"Agent command not found: {command}. Make sure the agent is installed.",
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def run_all_agents(
        self,
        scenario_path: Path,
        output_dir: Optional[Path] = None,
        agents: Optional[list[str]] = None,
        parallel: bool = False,
        max_workers: int = 3,
    ) -> list[dict]:
        """
        모든 Agent를 실행합니다.

        Args:
            scenario_path: 시나리오 JSON 파일 경로
            output_dir: 출력 디렉토리 (선택)
            agents: 실행할 Agent 목록 (선택, 기본: 모두)
            parallel: 병렬 실행 여부
            max_workers: 동시 실행 수 (병렬 모드 시)

        Returns:
            실행 결과 목록
        """
        target_agents = agents or list(AGENT_COMMANDS.keys())
        results = []

        if parallel and len(target_agents) > 1:
            # 병렬 실행: ThreadPoolExecutor + Semaphore
            semaphore = threading.Semaphore(max_workers)

            def run_with_semaphore(agent_id: str) -> dict:
                with semaphore:
                    return self.run_agent(agent_id, scenario_path, output_dir)

            with ThreadPoolExecutor(max_workers=len(target_agents)) as executor:
                futures = {
                    executor.submit(run_with_semaphore, aid): aid
                    for aid in target_agents
                }

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        agent_id = futures[future]
                        results.append({
                            "success": False,
                            "agent_id": agent_id,
                            "error": str(e),
                        })
        else:
            # 순차 실행
            for agent_id in target_agents:
                result = self.run_agent(agent_id, scenario_path, output_dir)
                results.append(result)

        return results
