"""
Analysis Stage Tools (4 tools)

RPISD's second stage: Rapid Analysis
- analyze_gap: Gap analysis
- analyze_performance: Performance analysis
- analyze_learner_characteristics: Learner characteristics analysis
- analyze_initial_task: Initial learning task analysis
"""

import json
from typing import List, Optional
from langchain_core.tools import tool

from shared.llm import create_chat_model, get_default_llm_config


def get_llm():
    return create_chat_model(get_default_llm_config())


def parse_json_response(content: str) -> dict:
    """Parse JSON from LLM response"""
    json_match = content
    if "```json" in content:
        json_match = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        json_match = content.split("```")[1].split("```")[0]
    return json.loads(json_match.strip())


@tool
def analyze_gap(
    learning_goals: List[str],
    current_state: Optional[str] = None,
    desired_state: Optional[str] = None,
) -> dict:
    """
    Perform Gap Analysis.

    Analyzes the gap between current state and desired state to
    derive educational needs.

    Args:
        learning_goals: List of learning goals
        current_state: Current state description
        desired_state: Desired state description

    Returns:
        Gap analysis results (gaps, root_causes, training_needs)
    """
    try:
        llm = get_llm()

        prompt = f"""You are an instructional designer specializing in rapid prototyping. Perform a Gap Analysis.

## Input Information
- Learning Goals: {json.dumps(learning_goals, ensure_ascii=False)}
- Current State: {current_state or "Not specified - assume general situation"}
- Desired State: {desired_state or "State where learning goals are achieved"}

## Analysis Items (Required)
1. **current_state**: Detailed description of current state
2. **desired_state**: Detailed description of desired state
3. **gaps**: Gaps between current and desired (minimum 3)
4. **root_causes**: Causes of gaps (minimum 3)
5. **training_needs**: Educational needs (minimum 3)

## Output Format (JSON)
```json
{{
  "current_state": "Detailed description of current state",
  "desired_state": "Detailed description of desired state",
  "gaps": [
    "Knowledge gap: Insufficient understanding of core concepts",
    "Skill gap: Lack of practical application ability",
    "Attitude gap: Lack of self-directed learning initiative"
  ],
  "root_causes": [
    "Lack of systematic training opportunities",
    "Insufficient practice and application experience",
    "Inadequate feedback system"
  ],
  "training_needs": [
    "Core concepts and principles education",
    "Hands-on workshop",
    "Continuous feedback and coaching"
  ]
}}
```

Output JSON only."""

        response = llm.invoke(prompt)
        return parse_json_response(response.content)
    except Exception:
        return _fallback_analyze_gap(learning_goals, current_state, desired_state)


def _fallback_analyze_gap(
    learning_goals: List[str],
    current_state: Optional[str] = None,
    desired_state: Optional[str] = None,
) -> dict:
    """Fallback function when LLM fails"""
    gaps = []
    for goal in learning_goals[:3]:
        gaps.append(f"Knowledge/skill gap: Insufficient competency related to {goal}")

    return {
        "current_state": current_state or "State with only basic knowledge related to learning goals",
        "desired_state": desired_state or "State where learning goals are achieved and applicable in practice",
        "gaps": gaps or [
            "Knowledge gap: Insufficient understanding of core concepts",
            "Skill gap: Lack of practical application ability",
            "Attitude gap: Lack of self-directed learning initiative",
        ],
        "root_causes": [
            "Lack of systematic training opportunities",
            "Insufficient practice and application experience",
            "Inadequate feedback system",
        ],
        "training_needs": [
            f"Training related to {goal}" for goal in learning_goals[:3]
        ] or ["Core concept education", "Hands-on workshop", "Feedback session"],
    }


@tool
def analyze_performance(
    learning_goals: List[str],
    performance_issues: Optional[List[str]] = None,
    organizational_context: Optional[str] = None,
) -> dict:
    """
    Perform Performance Analysis.

    Identifies causes of performance issues and determines
    appropriateness of training solutions.

    Args:
        learning_goals: List of learning goals
        performance_issues: List of performance issues
        organizational_context: Organizational context

    Returns:
        Performance analysis results (issues, causes, solutions, is_training_solution)
    """
    try:
        llm = get_llm()

        prompt = f"""You are an instructional designer specializing in rapid prototyping. Perform a Performance Analysis.

## Input Information
- Learning Goals: {json.dumps(learning_goals, ensure_ascii=False)}
- Performance Issues: {json.dumps(performance_issues, ensure_ascii=False) if performance_issues else "Not identified"}
- Organizational Context: {organizational_context or "General organization"}

## Analysis Items (Required)
1. **performance_issues**: Performance issues (minimum 3)
2. **causes**: Problem cause classification
3. **solutions**: Solution recommendations
4. **is_training_solution**: Determine if training solution is appropriate

## Output Format (JSON)
```json
{{
  "performance_issues": [
    "Decreased work processing speed",
    "Below quality standards",
    "Insufficient collaboration efficiency"
  ],
  "causes": {{
    "knowledge_skill": ["Lack of relevant knowledge", "Insufficient practical skills"],
    "motivation": ["Lack of motivation"],
    "environment": ["Inadequate support system", "Work overload"]
  }},
  "solutions": {{
    "training": ["Core competency training", "Hands-on workshop"],
    "non_training": ["Introduce mentoring program", "Improve work processes"]
  }},
  "is_training_solution": true,
  "recommendation": "Training solution is appropriate as main cause of performance issues is lack of knowledge/skills"
}}
```

Output JSON only."""

        response = llm.invoke(prompt)
        return parse_json_response(response.content)
    except Exception:
        return _fallback_analyze_performance(learning_goals, performance_issues)


def _fallback_analyze_performance(
    learning_goals: List[str],
    performance_issues: Optional[List[str]] = None,
) -> dict:
    """Fallback function when LLM fails"""
    return {
        "performance_issues": performance_issues or [
            "Decreased work processing speed",
            "Below quality standards",
            "Insufficient collaboration efficiency",
        ],
        "causes": {
            "knowledge_skill": ["Lack of relevant knowledge", "Insufficient practical skills"],
            "motivation": ["Lack of motivation"],
            "environment": ["Inadequate support system"],
        },
        "solutions": {
            "training": [f"Training related to {goal}" for goal in learning_goals[:2]],
            "non_training": ["Introduce mentoring program", "Improve work processes"],
        },
        "is_training_solution": True,
        "recommendation": "Training solution is appropriate as main cause of performance issues is lack of knowledge/skills",
    }


@tool
def analyze_learner_characteristics(
    target_audience: str,
    prior_knowledge: Optional[str] = None,
    learning_environment: Optional[str] = None,
    additional_context: Optional[str] = None,
) -> dict:
    """
    Perform learner characteristics analysis.

    In RPISD, analysis focuses on core characteristics
    for rapid prototyping.

    Args:
        target_audience: Target learners
        prior_knowledge: Prior knowledge level
        learning_environment: Learning environment
        additional_context: Additional context

    Returns:
        Learner characteristics analysis results
    """
    try:
        llm = get_llm()

        prompt = f"""You are an instructional designer specializing in rapid prototyping. Perform learner characteristics analysis.

## Input Information
- Target Audience: {target_audience}
- Prior Knowledge: {prior_knowledge or "Not specified"}
- Learning Environment: {learning_environment or "Not specified"}
- Additional Context: {additional_context or "None"}

## Analysis Items (Rapid Analysis - Focus on Core Characteristics)
1. **target_audience**: Target audience summary
2. **demographics**: Demographic characteristics
3. **prior_knowledge**: Prior knowledge level and experience
4. **learning_preferences**: Learning preferences (minimum 4)
5. **motivation**: Learning motivation
6. **challenges**: Expected challenges (minimum 3)

## Output Format (JSON)
```json
{{
  "target_audience": "{target_audience}",
  "demographics": {{
    "age_range": "Expected age range",
    "experience_level": "Experience level",
    "tech_literacy": "Technical familiarity"
  }},
  "prior_knowledge": "Detailed description of prior knowledge",
  "learning_preferences": [
    "Hands-on learning",
    "Visual materials",
    "Step-by-step guidance",
    "Immediate feedback"
  ],
  "motivation": "Description of learning motivation (intrinsic/extrinsic)",
  "challenges": [
    "Learning difficulties due to time constraints",
    "Difficulty understanding abstract concepts",
    "Identifying practical application methods"
  ]
}}
```

Output JSON only."""

        response = llm.invoke(prompt)
        return parse_json_response(response.content)
    except Exception:
        return _fallback_analyze_learner_characteristics(
            target_audience, prior_knowledge, learning_environment
        )


def _fallback_analyze_learner_characteristics(
    target_audience: str,
    prior_knowledge: Optional[str] = None,
    learning_environment: Optional[str] = None,
) -> dict:
    """Fallback function when LLM fails"""
    audience_lower = target_audience.lower()

    if "new" in audience_lower or "beginner" in audience_lower or "entry" in audience_lower:
        demographics = {"age_range": "20s-30s", "experience_level": "Beginner", "tech_literacy": "Average"}
        learning_preferences = ["Hands-on learning", "Step-by-step guidance", "Mentoring", "Immediate feedback"]
        motivation = "Strong motivation for quick adaptation and growth"
        challenges = ["Information overload", "Theory-practice gap", "Difficulty with self-directed learning"]
    elif "manager" in audience_lower or "leader" in audience_lower:
        demographics = {"age_range": "30s-50s", "experience_level": "Intermediate-Advanced", "tech_literacy": "Average-High"}
        learning_preferences = ["Case-based", "Discussion-focused", "Networking", "Efficient learning"]
        motivation = "Team performance improvement and leadership development"
        challenges = ["Time constraints", "Resistance to changing existing methods", "Intergenerational communication"]
    else:
        demographics = {"age_range": "Varies", "experience_level": "Mixed", "tech_literacy": "Average"}
        learning_preferences = ["Hands-on learning", "Visual materials", "Step-by-step guidance", "Immediate feedback"]
        motivation = "Motivation for competency development and performance improvement"
        challenges = ["Individual differences", "Maintaining motivation", "Lack of application opportunities"]

    return {
        "target_audience": target_audience,
        "demographics": demographics,
        "prior_knowledge": prior_knowledge or "Estimated to have basic level prior knowledge",
        "learning_preferences": learning_preferences,
        "motivation": motivation,
        "challenges": challenges,
    }


@tool
def analyze_initial_task(
    learning_goals: List[str],
    domain: Optional[str] = None,
    complexity_level: Optional[str] = None,
) -> dict:
    """
    Perform initial learning task analysis.

    Initial task analysis for rapid prototyping;
    detailed task analysis is performed after prototype development.

    Args:
        learning_goals: List of learning goals
        domain: Educational domain
        complexity_level: Complexity level

    Returns:
        Initial task analysis results (topics, subtopics, prerequisites, hierarchy)
    """
    try:
        llm = get_llm()

        prompt = f"""You are an instructional designer specializing in rapid prototyping. Perform initial learning task analysis.

## Input Information
- Learning Goals: {json.dumps(learning_goals, ensure_ascii=False)}
- Educational Domain: {domain or "General"}
- Complexity: {complexity_level or "Medium"}

## Analysis Items (Initial Analysis - Quick Assessment)
1. **main_topics**: Main topics (minimum 3)
2. **subtopics**: Subtopics (minimum 2 per topic)
3. **prerequisites**: Prerequisites (minimum 2)
4. **task_hierarchy**: Task hierarchy structure

## Output Format (JSON)
```json
{{
  "main_topics": [
    "Core concept understanding",
    "Basic skill acquisition",
    "Practical application"
  ],
  "subtopics": [
    "Core concept 1: Basic theory",
    "Core concept 2: Key principles",
    "Basic skill 1: Essential skills",
    "Basic skill 2: Applied skills",
    "Practical application 1: Case analysis",
    "Practical application 2: Practice assignments"
  ],
  "prerequisites": [
    "Basic terminology understanding",
    "Related background knowledge"
  ],
  "task_hierarchy": {{
    "level_1": "Basic knowledge and concepts",
    "level_2": "Principle understanding and application",
    "level_3": "Practical use and problem solving"
  }}
}}
```

Output JSON only."""

        response = llm.invoke(prompt)
        return parse_json_response(response.content)
    except Exception:
        return _fallback_analyze_initial_task(learning_goals, domain)


def _fallback_analyze_initial_task(
    learning_goals: List[str],
    domain: Optional[str] = None,
) -> dict:
    """Fallback function when LLM fails"""
    main_topics = learning_goals[:3] if len(learning_goals) >= 3 else learning_goals + ["Additional learning topic"]

    subtopics = []
    for topic in main_topics:
        subtopics.append(f"{topic} - Basic concepts")
        subtopics.append(f"{topic} - Application and practice")

    return {
        "main_topics": main_topics,
        "subtopics": subtopics[:6],
        "prerequisites": ["Basic terminology understanding", "Related background knowledge"],
        "task_hierarchy": {
            "level_1": "Basic knowledge and concepts",
            "level_2": "Principle understanding and application",
            "level_3": "Practical use and problem solving",
        },
    }
