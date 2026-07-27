"""
Skill-Tree 기반 학습자 역량 모델링

EduPlanner 논문의 Skill-Tree 구조를 교수설계 맥락으로 변환하여 적용합니다.

원본 (수학 능력):
- Numerical Calculation, Abstract Thinking, Logical Reasoning,
  Analogy Association, Spatial Imagination

변환 (학습자 역량):
- 사전 지식 수준, 학습 선호도, 동기 수준, 자기주도성, 기술 활용 능력
"""

from typing import Optional
from pydantic import BaseModel, Field


class SkillNode(BaseModel):
    """Skill-Tree의 개별 노드"""
    name: str = Field(..., description="Competency name")
    level: int = Field(..., ge=1, le=5, description="Competency level (1-5)")
    description: str = Field(..., description="Competency description")
    indicators: list[str] = Field(default_factory=list, description="Indicators by level")


class SkillTree(BaseModel):
    """학습자 역량 Skill-Tree"""

    prior_knowledge: SkillNode = Field(
        ...,
        description="Prior knowledge level"
    )
    learning_preference: SkillNode = Field(
        ...,
        description="Learning preference (visual/auditory/kinesthetic, etc.)"
    )
    motivation: SkillNode = Field(
        ...,
        description="Learning motivation level"
    )
    self_directedness: SkillNode = Field(
        ...,
        description="Self-directed learning ability"
    )
    tech_literacy: SkillNode = Field(
        ...,
        description="Technology/digital literacy"
    )

    def get_levels(self) -> list[int]:
        """모든 역량 수준을 리스트로 반환"""
        return [
            self.prior_knowledge.level,
            self.learning_preference.level,
            self.motivation.level,
            self.self_directedness.level,
            self.tech_literacy.level,
        ]

    def average_level(self) -> float:
        """평균 역량 수준 계산"""
        levels = self.get_levels()
        return sum(levels) / len(levels)

    def to_prompt_context(self) -> str:
        """프롬프트에 포함할 학습자 프로필 문자열 생성"""
        return f"""## Learner Competency Profile (Skill-Tree)

1. **Prior Knowledge Level**: {self.prior_knowledge.level}/5
   - {self.prior_knowledge.description}

2. **Learning Preference**: {self.learning_preference.level}/5
   - {self.learning_preference.description}

3. **Learning Motivation**: {self.motivation.level}/5
   - {self.motivation.description}

4. **Self-Directedness**: {self.self_directedness.level}/5
   - {self.self_directedness.description}

5. **Technology Literacy**: {self.tech_literacy.level}/5
   - {self.tech_literacy.description}

**Overall Level**: {self.average_level():.1f}/5
"""


class LearnerProfile(BaseModel):
    """학습자 프로필"""

    profile_id: str = Field(..., description="Profile ID")
    name: str = Field(..., description="Profile name (e.g. new hire, elementary school student)")
    skill_tree: SkillTree = Field(..., description="Competency Skill-Tree")
    characteristics: list[str] = Field(default_factory=list, description="Characteristics")
    challenges: list[str] = Field(default_factory=list, description="Anticipated difficulties")

    @classmethod
    def from_scenario(
        cls,
        target_audience: str,
        prior_knowledge: Optional[str] = None,
        learning_environment: Optional[str] = None,
    ) -> "LearnerProfile":
        """시나리오 정보로부터 학습자 프로필 생성"""

        # 기본값 설정 (추후 LLM으로 동적 생성 가능)
        skill_tree = cls._infer_skill_tree(target_audience, prior_knowledge)
        characteristics = cls._infer_characteristics(target_audience)
        challenges = cls._infer_challenges(target_audience, prior_knowledge)

        return cls(
            profile_id=f"LP-{hash(target_audience) % 10000:04d}",
            name=target_audience,
            skill_tree=skill_tree,
            characteristics=characteristics,
            challenges=challenges,
        )

    @staticmethod
    def _infer_skill_tree(
        target_audience: str,
        prior_knowledge: Optional[str] = None
    ) -> SkillTree:
        """대상자 정보로부터 Skill-Tree 추론"""

        # 간단한 규칙 기반 추론 (추후 LLM 기반으로 개선 가능)
        audience_lower = target_audience.lower()

        # 기본 수준
        base_levels = {
            "prior_knowledge": 3,
            "learning_preference": 3,
            "motivation": 3,
            "self_directedness": 3,
            "tech_literacy": 3,
        }

        # 대상자별 조정
        if "new hire" in audience_lower or "beginner" in audience_lower:
            base_levels["prior_knowledge"] = 2
            base_levels["motivation"] = 4
        elif "elementary" in audience_lower:
            base_levels["prior_knowledge"] = 2
            base_levels["self_directedness"] = 2
            base_levels["tech_literacy"] = 2
        elif "employee" in audience_lower or "adult" in audience_lower:
            base_levels["prior_knowledge"] = 3
            base_levels["motivation"] = 4
            base_levels["self_directedness"] = 4
            base_levels["tech_literacy"] = 4
        elif "expert" in audience_lower or "advanced" in audience_lower:
            base_levels["prior_knowledge"] = 5
            base_levels["self_directedness"] = 5

        return SkillTree(
            prior_knowledge=SkillNode(
                name="Prior Knowledge",
                level=base_levels["prior_knowledge"],
                description=prior_knowledge or "Has basic knowledge of the relevant field",
                indicators=[
                    "Lv1: No related knowledge",
                    "Lv2: Understands basic concepts",
                    "Lv3: Intermediate level",
                    "Lv4: Advanced level",
                    "Lv5: Expert level",
                ],
            ),
            learning_preference=SkillNode(
                name="Learning Preference",
                level=base_levels["learning_preference"],
                description="Receptiveness to a variety of learning modes",
                indicators=[
                    "Lv1: Prefers only one specific mode",
                    "Lv2: Limited receptiveness",
                    "Lv3: Average",
                    "Lv4: Flexible receptiveness",
                    "Lv5: Adapts to all modes",
                ],
            ),
            motivation=SkillNode(
                name="Learning Motivation",
                level=base_levels["motivation"],
                description="Level of intrinsic/extrinsic motivation for learning",
                indicators=[
                    "Lv1: Lacks motivation",
                    "Lv2: Mainly extrinsic motivation",
                    "Lv3: Average",
                    "Lv4: High motivation",
                    "Lv5: Very high intrinsic motivation",
                ],
            ),
            self_directedness=SkillNode(
                name="Self-Directedness",
                level=base_levels["self_directedness"],
                description="Ability to plan and carry out learning independently",
                indicators=[
                    "Lv1: Requires full guidance",
                    "Lv2: Requires partial guidance",
                    "Lv3: Average",
                    "Lv4: Mostly self-directed",
                    "Lv5: Fully self-directed",
                ],
            ),
            tech_literacy=SkillNode(
                name="Technology Literacy",
                level=base_levels["tech_literacy"],
                description="Ability to use digital tools and technology",
                indicators=[
                    "Lv1: Struggles with technology",
                    "Lv2: Can use basic features",
                    "Lv3: Average",
                    "Lv4: Proficient use",
                    "Lv5: Expert use",
                ],
            ),
        )

    @staticmethod
    def _infer_characteristics(target_audience: str) -> list[str]:
        """대상자 특성 추론"""
        audience_lower = target_audience.lower()
        characteristics = []

        if "new hire" in audience_lower:
            characteristics.extend([
                "Needs to adapt to organizational culture",
                "Desire for rapid growth",
                "Values practical application on the job",
            ])
        elif "elementary" in audience_lower:
            characteristics.extend([
                "Short attention span",
                "Prefers game/play-based learning",
                "Visual materials are effective",
            ])
        elif "employee" in audience_lower:
            characteristics.extend([
                "Has time constraints",
                "Values practical application on the job",
                "Prefers efficient learning",
            ])

        return characteristics

    @staticmethod
    def _infer_challenges(
        target_audience: str,
        prior_knowledge: Optional[str] = None
    ) -> list[str]:
        """예상 어려움 추론"""
        challenges = []
        audience_lower = target_audience.lower()

        if "new hire" in audience_lower or "beginner" in audience_lower:
            challenges.append("May lack understanding of basic concepts")
        if "elementary" in audience_lower:
            challenges.append("Difficulty understanding abstract concepts")
            challenges.append("Difficulty concentrating for long periods")
        if prior_knowledge and "none" in prior_knowledge:
            challenges.append("Prerequisite learning required")

        return challenges


# 사전 정의된 학습자 프로필 템플릿
PROFILE_TEMPLATES = {
    "beginner": LearnerProfile(
        profile_id="TPL-BEG",
        name="Beginner Learner",
        skill_tree=SkillTree(
            prior_knowledge=SkillNode(
                name="Prior Knowledge", level=2,
                description="Understands only basic concepts", indicators=[]
            ),
            learning_preference=SkillNode(
                name="Learning Preference", level=3,
                description="Prefers visual materials", indicators=[]
            ),
            motivation=SkillNode(
                name="Learning Motivation", level=3,
                description="Average level of motivation", indicators=[]
            ),
            self_directedness=SkillNode(
                name="Self-Directedness", level=2,
                description="Requires guidance", indicators=[]
            ),
            tech_literacy=SkillNode(
                name="Technology Literacy", level=3,
                description="Can use basic features", indicators=[]
            ),
        ),
        characteristics=["Needs to start from the basics", "Needs step-by-step guidance"],
        challenges=["Difficulty understanding complex concepts"],
    ),
    "intermediate": LearnerProfile(
        profile_id="TPL-INT",
        name="Intermediate Learner",
        skill_tree=SkillTree(
            prior_knowledge=SkillNode(
                name="Prior Knowledge", level=3,
                description="Has intermediate-level knowledge", indicators=[]
            ),
            learning_preference=SkillNode(
                name="Learning Preference", level=4,
                description="Receptive to various modes", indicators=[]
            ),
            motivation=SkillNode(
                name="Learning Motivation", level=4,
                description="Strong willingness to learn", indicators=[]
            ),
            self_directedness=SkillNode(
                name="Self-Directedness", level=4,
                description="Mostly self-directed", indicators=[]
            ),
            tech_literacy=SkillNode(
                name="Technology Literacy", level=4,
                description="Proficient use", indicators=[]
            ),
        ),
        characteristics=["Capable of advanced learning", "Self-directed"],
        challenges=["Leap to advanced concepts"],
    ),
    "advanced": LearnerProfile(
        profile_id="TPL-ADV",
        name="Advanced Learner",
        skill_tree=SkillTree(
            prior_knowledge=SkillNode(
                name="Prior Knowledge", level=5,
                description="Expert level", indicators=[]
            ),
            learning_preference=SkillNode(
                name="Learning Preference", level=5,
                description="Adapts to all modes", indicators=[]
            ),
            motivation=SkillNode(
                name="Learning Motivation", level=5,
                description="Very high intrinsic motivation", indicators=[]
            ),
            self_directedness=SkillNode(
                name="Self-Directedness", level=5,
                description="Fully self-directed", indicators=[]
            ),
            tech_literacy=SkillNode(
                name="Technology Literacy", level=5,
                description="Expert use", indicators=[]
            ),
        ),
        characteristics=["Deepening expertise", "Capable of a leadership role"],
        challenges=["Needs new challenges"],
    ),
}
