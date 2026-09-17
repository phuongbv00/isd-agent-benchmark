"""
Goal & Analysis Tools (Steps 1-3)

Dick & Carey Model's first three steps:
1. Instructional Goal Setting
2. Instructional Analysis
3. Entry Behaviors & Context Analysis
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


# ========== Step 1: Instructional Goal Setting ==========
@tool
def set_instructional_goal(
    learning_goals: List[str],
    target_audience: str,
    current_state: Optional[str] = None,
    desired_state: Optional[str] = None,
) -> dict:
    """
    Set instructional goals. (Dick & Carey Step 1)

    Instructional goals state what learners should be able to perform after instruction.
    Analyzes the gap between current state and desired state based on needs analysis results.
    Includes sub-items A-1 through A-4.

    Args:
        learning_goals: List of learning goals
        target_audience: Target learners
        current_state: Current state description (optional)
        desired_state: Desired state description (optional)

    Returns:
        Instructional goal setting results (goal_statement, target_domain, performance_gap, needs_analysis)
    """
    try:
        llm = get_llm()

        prompt = f"""You are an expert in the Dick & Carey model. Set the Instructional Goal based on the following information.

## Input Information
- Learning Goals: {json.dumps(learning_goals, ensure_ascii=False)}
- Target Audience: {target_audience}
- Current State: {current_state or "Not specified"}
- Desired State: {desired_state or "Not specified"}

## Dick & Carey's Instructional Goal Setting Principles
1. Clearly state what learners should be able to perform after instruction
2. Identify target domain (cognitive/affective/psychomotor)
3. Analyze performance gap between current and desired state
4. Needs analysis and priority determination (A-1 through A-4)

## Required Elements
1. goal_statement: Specific and measurable instructional goal statement (1-2 sentences)
2. target_domain: Primary learning domain (cognitive, affective, psychomotor)
3. current_state: Current learner state analysis
4. desired_state: Target learner state
5. performance_gap: Performance gap analysis (2-3 sentences)
6. needs_analysis: Needs analysis results (A-1 through A-4)

## Output Format (JSON)
```json
{{
  "goal_statement": "After instruction, learners will be able to [specific performance].",
  "target_domain": "cognitive",
  "current_state": "Currently learners only understand basic concepts",
  "desired_state": "Learners can independently apply knowledge in practice",
  "performance_gap": "A gap exists between theoretical knowledge and practical application ability. Particularly lacking in problem-solving skills in real situations.",
  "needs_analysis": {{
    "gap_analysis": [
      "Knowledge gap between current and target levels",
      "Performance gap due to lack of practical application experience"
    ],
    "root_causes": [
      "Lack of systematic training opportunities",
      "Insufficient practical exercise environment",
      "Absence of feedback and coaching"
    ],
    "training_needs": [
      "Core concepts and principles learning",
      "Practical application exercises",
      "Problem-solving skill development"
    ],
    "non_training_solutions": [
      "Provide work manuals and guides",
      "Establish mentoring system",
      "Improve performance management system"
    ],
    "priority_matrix": {{
      "high_impact_high_urgency": ["Core concept learning", "Practical application exercises"],
      "high_impact_low_urgency": ["Advanced learning", "Advanced skill acquisition"],
      "low_impact_high_urgency": ["Basic review"],
      "low_impact_low_urgency": ["Reference material provision"]
    }},
    "recommendation": "Educational solutions are most effective, combined with non-instructional support (manuals, mentoring) for sustained performance improvement"
  }}
}}
```

Output JSON only."""

        response = llm.invoke(prompt)
        return parse_json_response(response.content)
    except Exception:
        return _fallback_set_instructional_goal(learning_goals, target_audience, current_state, desired_state)


def _fallback_set_instructional_goal(
    learning_goals: List[str],
    target_audience: str,
    current_state: Optional[str] = None,
    desired_state: Optional[str] = None,
) -> dict:
    """Fallback function when LLM fails"""
    goal_text = ", ".join(learning_goals[:2]) if learning_goals else "learning content"

    return {
        "goal_statement": f"After instruction, {target_audience} will be able to understand and apply {goal_text} in practice.",
        "target_domain": "cognitive",
        "current_state": current_state or "Basic knowledge level",
        "desired_state": desired_state or "Level capable of independent job performance",
        "performance_gap": "Currently has theoretical knowledge but lacks practical application experience. Systematic learning is needed to bridge this gap.",
        "needs_analysis": {
            "gap_analysis": [
                "Knowledge gap between current and target levels",
                "Performance gap due to lack of practical application experience",
            ],
            "root_causes": [
                "Lack of systematic training opportunities",
                "Insufficient practical exercise environment",
                "Absence of feedback and coaching",
            ],
            "training_needs": [
                "Core concepts and principles learning",
                "Practical application exercises",
                "Problem-solving skill development",
            ],
            "non_training_solutions": [
                "Provide work manuals and guides",
                "Establish mentoring system",
                "Improve performance management system",
            ],
            "priority_matrix": {
                "high_impact_high_urgency": ["Core concept learning", "Practical application exercises"],
                "high_impact_low_urgency": ["Advanced learning", "Advanced skill acquisition"],
                "low_impact_high_urgency": ["Basic review"],
                "low_impact_low_urgency": ["Reference material provision"],
            },
            "recommendation": "Educational solutions are most effective, combined with non-instructional support (manuals, mentoring) for sustained performance improvement",
        },
    }


# ========== Step 2: Instructional Analysis ==========
@tool
def analyze_instruction(
    instructional_goal: str,
    domain: Optional[str] = None,
    learning_goals: Optional[List[str]] = None,
) -> dict:
    """
    Conduct instructional analysis. (Dick & Carey Step 2)

    Analyzes sub-skills and procedures needed to achieve instructional goal.
    Identifies task type (procedural, hierarchical, combination, cluster) and constructs skill hierarchy.

    Args:
        instructional_goal: Instructional goal statement
        domain: Educational domain (optional)
        learning_goals: Detailed learning goals list (optional)

    Returns:
        Instructional analysis results (task_type, sub_skills, skill_hierarchy, entry_skills)
    """
    try:
        llm = get_llm()

        prompt = f"""You are an expert in the Dick & Carey model. Conduct an Instructional Analysis for the following instructional goal.

## Input Information
- Instructional Goal: {instructional_goal}
- Domain: {domain or "General"}
- Detailed Goals: {json.dumps(learning_goals, ensure_ascii=False) if learning_goals else "Not specified"}

## Dick & Carey's Instructional Analysis Principles
1. Task Type Classification:
   - Procedural: Tasks where sequential order is important
   - Hierarchical: Tasks requiring prerequisite skills
   - Combination: Procedural + Hierarchical
   - Cluster: Collection of independent skills
2. Sub-skills Derivation: Detailed skills needed for goal achievement
3. Skill Hierarchy Construction: Identify prerequisite relationships

## Required Elements
1. task_type: Task type (procedural, hierarchical, combination, cluster)
2. sub_skills: Sub-skills **minimum 5** (id, description, type, prerequisites)
3. skill_hierarchy: Skill hierarchy structure
4. entry_skills: Entry-level skills **minimum 2**

## Additional Required Elements
4. review_summary: A-10 Task analysis review/summary (summarize task analysis results and derive implications for instructional design)

## Output Format (JSON)
**Important**:
- Use skill_name field instead of id field.
- Use actual description text in prerequisites and skill_hierarchy.
- Must include review_summary field with task analysis review.

```json
{{
  "task_type": "combination",
  "sub_skills": [
    {{
      "skill_name": "Core concept understanding",
      "description": "Can explain core concepts",
      "type": "intellectual skill",
      "prerequisites": []
    }},
    {{
      "skill_name": "Case analysis",
      "description": "Can analyze related cases",
      "type": "intellectual skill",
      "prerequisites": ["Can explain core concepts"]
    }},
    {{
      "skill_name": "Problem diagnosis",
      "description": "Can diagnose problem situations",
      "type": "cognitive strategy",
      "prerequisites": ["Can explain core concepts", "Can analyze related cases"]
    }},
    {{
      "skill_name": "Solution design",
      "description": "Can design solutions",
      "type": "intellectual skill",
      "prerequisites": ["Can diagnose problem situations"]
    }},
    {{
      "skill_name": "Result evaluation",
      "description": "Can evaluate results and make improvements",
      "type": "cognitive strategy",
      "prerequisites": ["Can design solutions"]
    }}
  ],
  "skill_hierarchy": {{
    "level_1": ["Can explain core concepts"],
    "level_2": ["Can analyze related cases"],
    "level_3": ["Can diagnose problem situations"],
    "level_4": ["Can design solutions"],
    "level_5": ["Can evaluate results and make improvements"]
  }},
  "entry_skills": [
    "Basic terminology understanding",
    "Basic computer literacy"
  ],
  "review_summary": "Task analysis resulted in 5 sub-skills with a combination (procedural + hierarchical) structure. Achieving the instructional goal requires step-by-step learning from basic concept understanding to result evaluation. Entry skills require basic terminology understanding and computer literacy, enabling systematic instructional design."
}}
```

Output JSON only."""

        response = llm.invoke(prompt)
        return parse_json_response(response.content)
    except Exception:
        return _fallback_analyze_instruction(instructional_goal, domain, learning_goals)


def _fallback_analyze_instruction(
    instructional_goal: str,
    domain: Optional[str] = None,
    learning_goals: Optional[List[str]] = None,
) -> dict:
    """Fallback function when LLM fails"""
    goals = learning_goals or [instructional_goal]

    # Sub-skill description texts
    desc_1 = f"Understanding basic concepts related to {goals[0]}"
    desc_2 = "Analyzing related cases"
    desc_3 = "Diagnosing problem situations"
    desc_4 = "Designing and applying solutions"
    desc_5 = "Evaluating results and making improvements"

    sub_skills = [
        {"skill_name": "Core concept understanding", "description": desc_1, "type": "intellectual skill", "prerequisites": []},
        {"skill_name": "Case analysis", "description": desc_2, "type": "intellectual skill", "prerequisites": [desc_1]},
        {"skill_name": "Problem diagnosis", "description": desc_3, "type": "cognitive strategy", "prerequisites": [desc_1, desc_2]},
        {"skill_name": "Solution design", "description": desc_4, "type": "intellectual skill", "prerequisites": [desc_3]},
        {"skill_name": "Result evaluation", "description": desc_5, "type": "cognitive strategy", "prerequisites": [desc_4]},
    ]

    return {
        "task_type": "combination",
        "sub_skills": sub_skills,
        "skill_hierarchy": {
            "level_1": [desc_1],
            "level_2": [desc_2],
            "level_3": [desc_3],
            "level_4": [desc_4],
            "level_5": [desc_5],
        },
        "entry_skills": ["Basic terminology understanding", "Basic learning ability"],
        "review_summary": f"Task analysis resulted in 5 sub-skills with a combination (procedural + hierarchical) structure. Achieving {instructional_goal} requires step-by-step learning from basic concept understanding to result evaluation.",
    }


# ========== Step 3: Entry Behaviors & Context Analysis ==========
@tool
def analyze_entry_behaviors(
    target_audience: str,
    prior_knowledge: Optional[str] = None,
    entry_skills: Optional[List[str]] = None,
) -> dict:
    """
    Analyze learner entry behaviors. (Dick & Carey Step 3 - Learner Analysis)

    Analyzes knowledge, skills, and attitudes learners should possess before instruction begins.

    Args:
        target_audience: Target learners
        prior_knowledge: Prior knowledge level (optional)
        entry_skills: Entry skills list (optional)

    Returns:
        Learner analysis results (entry_behaviors, characteristics, learning_preferences, motivation)
    """
    try:
        llm = get_llm()

        prompt = f"""You are an expert in the Dick & Carey model. Analyze the learner's Entry Behaviors.

## Input Information
- Target Audience: {target_audience}
- Prior Knowledge: {prior_knowledge or "Not specified"}
- Entry Skills: {json.dumps(entry_skills, ensure_ascii=False) if entry_skills else "Not specified"}

## Dick & Carey's Learner Analysis Principles
1. Entry Behaviors: Skills learners should possess before instruction begins
2. Learner Characteristics: General and academic characteristics
3. Learning Preferences: Preferred learning methods
4. Motivation: Level of learning motivation

## Required Elements
1. target_audience: Target learners
2. entry_behaviors: Entry behaviors **minimum 3**
3. characteristics: Learner characteristics **minimum 5**
4. prior_knowledge: Prior knowledge analysis
5. learning_preferences: Learning preferences **minimum 4**
6. motivation: Motivation level analysis (2-3 sentences)

## Output Format (JSON)
```json
{{
  "target_audience": "{target_audience}",
  "entry_behaviors": [
    "Understands basic terminology",
    "Has foundational knowledge in related field",
    "Possesses self-directed learning ability"
  ],
  "characteristics": [
    "Active attitude toward acquiring new knowledge",
    "High interest in practical application",
    "Preference for collaborative learning with peers",
    "Expectation for immediate feedback",
    "Familiar with digital tools"
  ],
  "prior_knowledge": "Has basic theoretical knowledge but lacks practical experience",
  "learning_preferences": [
    "Hands-on experiential learning",
    "Step-by-step structured guidance",
    "Case-based learning",
    "Collaborative learning activities"
  ],
  "motivation": "Strong intrinsic motivation for competency improvement. Clear purpose to acquire knowledge applicable to actual work."
}}
```

Output JSON only."""

        response = llm.invoke(prompt)
        return parse_json_response(response.content)
    except Exception:
        return _fallback_analyze_entry_behaviors(target_audience, prior_knowledge, entry_skills)


def _fallback_analyze_entry_behaviors(
    target_audience: str,
    prior_knowledge: Optional[str] = None,
    entry_skills: Optional[List[str]] = None,
) -> dict:
    """Fallback function when LLM fails"""
    return {
        "target_audience": target_audience,
        "entry_behaviors": entry_skills or [
            "Basic terminology understanding",
            "Foundational knowledge in related field",
            "Self-directed learning ability",
        ],
        "characteristics": [
            "Active in acquiring new knowledge",
            "High interest in practical application",
            "Preference for collaborative learning",
            "Expectation for immediate feedback",
            "Familiar with digital tools",
        ],
        "prior_knowledge": prior_knowledge or "Has basic knowledge, needs advanced learning",
        "learning_preferences": ["Hands-on focused", "Step-by-step guidance", "Case-based", "Collaborative activities"],
        "motivation": "Has intrinsic motivation for competency improvement, with a clear purpose to apply in actual work.",
    }


@tool
def analyze_context(
    learning_environment: str,
    duration: str,
    performance_context: Optional[str] = None,
    class_size: Optional[int] = None,
    resources: Optional[List[str]] = None,
) -> dict:
    """
    Analyze learning and performance context. (Dick & Carey Step 3 - Context Analysis)

    Analyzes the environment where learning takes place and the performance environment where learning outcomes are applied.

    Args:
        learning_environment: Learning environment (e.g., "online", "in-person", "blended")
        duration: Learning duration
        performance_context: Performance environment - where learning outcomes are applied (optional)
        class_size: Number of learners (optional)
        resources: Available resources list (optional)

    Returns:
        Context analysis results (performance_context, learning_context, constraints, resources, technical_requirements)
    """
    try:
        llm = get_llm()

        prompt = f"""You are an expert in the Dick & Carey model. Analyze the learning and performance context.

## Input Information
- Learning Environment: {learning_environment}
- Duration: {duration}
- Performance Context: {performance_context or "Not specified"}
- Number of Learners: {class_size or "Not specified"}
- Available Resources: {json.dumps(resources, ensure_ascii=False) if resources else "Not specified"}

## Dick & Carey's Context Analysis Principles
1. Performance Context: Environment where learning outcomes are actually applied
2. Learning Context: Environment where instruction takes place
3. Similarity between two environments affects Transfer

## Required Elements
1. performance_context: Performance context analysis (2-3 sentences)
2. learning_context: Learning context analysis (2-3 sentences)
3. constraints: Constraints **minimum 3**
4. resources: Available resources **minimum 3**
5. technical_requirements: Technical requirements **minimum 2**

## Output Format (JSON)
```json
{{
  "performance_context": "Learners will apply acquired skills in actual work settings. Problem-solving in various situations is required.",
  "learning_context": "Learning will be conducted in {learning_environment} environment for {duration}. Individual and group activities will be combined.",
  "constraints": [
    "Limited learning time",
    "Varying skill levels among learners",
    "Technical environment constraints"
  ],
  "resources": [
    "Learning Management System (LMS)",
    "Multimedia materials",
    "Practice environment",
    "Expert support"
  ],
  "technical_requirements": [
    "Stable internet connection",
    "Device with access to learning platform"
  ]
}}
```

Output JSON only."""

        response = llm.invoke(prompt)
        return parse_json_response(response.content)
    except Exception:
        return _fallback_analyze_context(learning_environment, duration, performance_context, class_size, resources)


def _fallback_analyze_context(
    learning_environment: str,
    duration: str,
    performance_context: Optional[str] = None,
    class_size: Optional[int] = None,
    resources: Optional[List[str]] = None,
) -> dict:
    """Fallback function when LLM fails"""
    env_lower = learning_environment.lower()

    constraints = ["Limited learning time", "Varying skill levels among learners"]
    technical_requirements = ["Learning platform access"]
    available_resources = resources or []

    if "online" in env_lower:
        constraints.append("Dependency on technical environment")
        technical_requirements = ["Stable internet connection", "Video conferencing tools"]
        if not available_resources:
            available_resources = ["LMS", "Video conferencing", "Recorded materials", "Online practice environment"]
    elif "in-person" in env_lower or "face-to-face" in env_lower:
        constraints.append("Physical space constraints")
        technical_requirements = ["Projector/screen", "Classroom reservation"]
        if not available_resources:
            available_resources = ["Textbooks", "Presentations", "Practice materials", "Whiteboard"]
    else:
        if not available_resources:
            available_resources = ["Learning materials", "Practice environment", "Feedback system"]

    if class_size and class_size > 30:
        constraints.append("Large group management needed")

    return {
        "performance_context": performance_context or "Learning content will be applied in actual work settings",
        "learning_context": f"Learning will be conducted in {learning_environment} environment for {duration}",
        "constraints": constraints,
        "resources": available_resources,
        "technical_requirements": technical_requirements,
    }
