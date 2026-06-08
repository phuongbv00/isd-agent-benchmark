"""
비교 리포터

Agent 평가 결과를 다양한 형식으로 리포팅합니다.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional


def _addie_score(r: dict) -> float:
    """Return ADDIE score across old and multi-judge result schemas."""
    for value in (
        r.get("addie_median"),
        r.get("addie_score"),
        r.get("output_score"),
        r.get("scores", {}).get("addie_median"),
    ):
        if value is not None:
            return value
    return 0


def _score_cell(scores: dict, key: str) -> str:
    value = scores.get(key)
    return f"{value:.1f}" if value is not None else "N/A"


class ComparisonReporter:
    """비교 리포터"""

    def __init__(self):
        pass

    def generate_markdown(
        self,
        comparison_result: dict,
        scenario: Optional[dict] = None,
        output_path: Optional[Path] = None,
    ) -> str:
        """
        마크다운 형식 리포트 생성

        Args:
            comparison_result: compare_agents() 결과
            scenario: 시나리오 정보 (선택)
            output_path: 저장 경로 (선택)

        Returns:
            마크다운 문자열
        """
        lines = []

        # 헤더
        lines.append("# 교수설계 Agent 비교 평가 리포트")
        lines.append("")
        lines.append(f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # 시나리오 정보
        if scenario:
            lines.append("## 평가 시나리오")
            lines.append("")
            lines.append(f"- **제목:** {scenario.get('title', 'N/A')}")
            lines.append(f"- **ID:** {scenario.get('scenario_id', 'N/A')}")
            context = scenario.get("context", {})
            lines.append(f"- **대상:** {context.get('target_audience', 'N/A')}")
            lines.append(f"- **시간:** {context.get('duration', 'N/A')}")
            lines.append(f"- **환경:** {context.get('learning_environment', 'N/A')}")
            lines.append("")

        # 순위 테이블
        lines.append("## 종합 순위")
        lines.append("")
        lines.append("| 순위 | Agent | 총점 | ADDIE 점수 | Trajectory 점수 |")
        lines.append("|------|-------|------|-----------|----------------|")

        rankings = comparison_result.get("rankings", [])
        for r in rankings:
            traj = r.get("trajectory_score")
            traj_str = f"{traj:.1f}" if traj is not None else "N/A"
            addie_score = _addie_score(r)
            lines.append(
                f"| {r['rank']} | {r['agent_id']} | "
                f"{r['total_score']:.1f} | {addie_score:.1f} | {traj_str} |"
            )

        lines.append("")

        # ADDIE 단계별 점수
        lines.append("## ADDIE 단계별 점수 (%)")
        lines.append("")
        lines.append("| Agent | Analysis | Design | Development | Implementation | Evaluation |")
        lines.append("|-------|----------|--------|-------------|----------------|------------|")

        for r in rankings:
            phase_scores = r.get("phase_scores", {})
            if not phase_scores:
                # 기존 details에서 추출 시도
                details = r.get("details", {})
                phases = details.get("phases", {})
                phase_scores = {
                    k: v.get("percentage", 0) for k, v in phases.items()
                }

            lines.append(
                f"| {r['agent_id']} | "
                f"{_score_cell(phase_scores, 'analysis')} | "
                f"{_score_cell(phase_scores, 'design')} | "
                f"{_score_cell(phase_scores, 'development')} | "
                f"{_score_cell(phase_scores, 'implementation')} | "
                f"{_score_cell(phase_scores, 'evaluation')} |"
            )

        lines.append("")

        # ADDIE 상세 항목별 점수 (선택적)
        lines.append("## ADDIE 항목별 점수 (0-10)")
        lines.append("")
        lines.append("| Agent | A1 | A2 | A3 | D1 | D2 | D3 | Dev1 | Dev2 | I1 | I2 | E1 | E2 | E3 |")
        lines.append("|-------|----|----|----|----|----|----|------|------|----|----|----|-----|----|")

        for r in rankings:
            details = r.get("details", {})
            phases = details.get("phases", {})

            # 항목별 점수 추출
            item_scores = {}
            for phase_name, phase_data in phases.items():
                items = phase_data.get("items", [])
                for item in items:
                    item_id = item.get("item_id", "")
                    score = item.get("score", 0)
                    item_scores[item_id] = score

            lines.append(
                f"| {r['agent_id']} | "
                f"{_score_cell(item_scores, 'A1')} | "
                f"{_score_cell(item_scores, 'A2')} | "
                f"{_score_cell(item_scores, 'A3')} | "
                f"{_score_cell(item_scores, 'D1')} | "
                f"{_score_cell(item_scores, 'D2')} | "
                f"{_score_cell(item_scores, 'D3')} | "
                f"{_score_cell(item_scores, 'Dev1')} | "
                f"{_score_cell(item_scores, 'Dev2')} | "
                f"{_score_cell(item_scores, 'I1')} | "
                f"{_score_cell(item_scores, 'I2')} | "
                f"{_score_cell(item_scores, 'E1')} | "
                f"{_score_cell(item_scores, 'E2')} | "
                f"{_score_cell(item_scores, 'E3')} |"
            )

        lines.append("")

        # 궤적 점수 (있는 경우)
        has_trajectory = any(
            r.get("trajectory_score") is not None for r in rankings
        )

        if has_trajectory:
            lines.append("## 도구 사용 평가 점수 (BFCL 기반)")
            lines.append("")
            lines.append("| Agent | 도구 정확성 | 인자 정확성 | 중복 회피 | 결과 활용도 |")
            lines.append("|-------|------------|------------|----------|------------|")

            for r in rankings:
                traj_details = r.get("trajectory_details")
                if traj_details:
                    lines.append(
                        f"| {r['agent_id']} | "
                        f"{traj_details.get('tool_correctness', 0):.1f} | "
                        f"{traj_details.get('argument_accuracy', 0):.1f} | "
                        f"{traj_details.get('redundancy_avoidance', 0):.1f} | "
                        f"{traj_details.get('result_utilization', 0):.1f} |"
                    )
                else:
                    lines.append(f"| {r['agent_id']} | N/A | N/A | N/A | N/A |")

            lines.append("")

        # 강점/개선점 (있는 경우)
        lines.append("## 평가 요약")
        lines.append("")

        for r in rankings:
            details = r.get("details", {})
            strengths = details.get("strengths", [])
            improvements = details.get("improvements", [])
            overall = details.get("overall_assessment", "")

            lines.append(f"### {r['agent_id']}")
            lines.append("")

            if overall:
                lines.append(f"**종합 평가:** {overall}")
                lines.append("")

            if strengths:
                lines.append("**강점:**")
                for s in strengths[:3]:  # 상위 3개
                    lines.append(f"- {s}")
                lines.append("")

            if improvements:
                lines.append("**개선점:**")
                for i in improvements[:3]:  # 상위 3개
                    lines.append(f"- {i}")
                lines.append("")

        # 결론
        lines.append("## 결론")
        lines.append("")
        best = comparison_result.get("best_agent", "N/A")
        lines.append(f"**최고 성능 Agent:** {best}")
        lines.append("")

        report = "\n".join(lines)

        # 파일 저장 (선택)
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report)

        return report

    def generate_json(
        self,
        comparison_result: dict,
        scenario: Optional[dict] = None,
        output_path: Optional[Path] = None,
    ) -> dict:
        """
        JSON 형식 리포트 생성

        Args:
            comparison_result: compare_agents() 결과
            scenario: 시나리오 정보 (선택)
            output_path: 저장 경로 (선택)

        Returns:
            JSON 딕셔너리
        """
        import json

        report = {
            "generated_at": datetime.now().isoformat(),
            "scenario": scenario,
            "comparison": comparison_result,
        }

        # 파일 저장 (선택)
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)

        return report

    def print_summary(self, comparison_result: dict) -> None:
        """콘솔에 요약 출력"""
        print("\n" + "=" * 60)
        print("교수설계 Agent 비교 평가 결과")
        print("=" * 60)

        rankings = comparison_result.get("rankings", [])

        for r in rankings:
            print(f"\n#{r['rank']} {r['agent_id']}")
            print(f"   총점: {r['total_score']:.1f}/100")

            addie_score = _addie_score(r)
            print(f"   ADDIE: {addie_score:.1f}")

            if r.get("trajectory_score") is not None:
                print(f"   Trajectory: {r['trajectory_score']:.1f}")

            # 단계별 점수 표시
            phase_scores = r.get("phase_scores", {})
            if phase_scores:
                phases_str = " | ".join([
                    f"A:{phase_scores.get('analysis', 0):.0f}%",
                    f"D:{phase_scores.get('design', 0):.0f}%",
                    f"Dev:{phase_scores.get('development', 0):.0f}%",
                    f"I:{phase_scores.get('implementation', 0):.0f}%",
                    f"E:{phase_scores.get('evaluation', 0):.0f}%",
                ])
                print(f"   단계별: {phases_str}")

        print("\n" + "-" * 60)
        print(f"최고 성능: {comparison_result.get('best_agent', 'N/A')}")
        print("=" * 60 + "\n")
