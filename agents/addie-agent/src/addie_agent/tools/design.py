"""
Design Stage Tools

ADDIE's second stage: Learning objectives, assessment plan, instructional strategy design
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


BLOOM_VERBS = {
    "Remember": ["define", "list", "name", "recognize", "recall"],
    "Understand": ["explain", "summarize", "interpret", "classify", "compare"],
    "Apply": ["apply", "demonstrate", "use", "execute", "implement"],
    "Analyze": ["analyze", "distinguish", "organize", "critique", "examine"],
    "Evaluate": ["evaluate", "judge", "justify", "critique", "recommend"],
    "Create": ["design", "develop", "generate", "construct", "create"],
}


@tool
def design_objectives(
    learning_goals: list[str],
    target_audience: str,
    difficulty: Optional[str] = None,
) -> list[dict]:
    """
    Design learning objectives based on Bloom's Taxonomy.

    Args:
        learning_goals: Original learning goals list
        target_audience: Target learners
        difficulty: Difficulty level (optional)

    Returns:
        Learning objectives list (minimum 5, distributed across Bloom's levels)
    """
    try:
        llm = get_llm()

        prompt = f"""You are an instructional design expert with 20 years of experience. Design measurable learning objectives based on Bloom's Taxonomy.

## Input Information
- Original Learning Goals: {json.dumps(learning_goals, ensure_ascii=False)}
- Target Audience: {target_audience}
- Difficulty: {difficulty or "medium"}

## Bloom's Taxonomy Verbs
- Remember: define, list, name, recognize, recall
- Understand: explain, summarize, interpret, classify, compare
- Apply: apply, demonstrate, use, execute, implement
- Analyze: analyze, distinguish, organize, critique, examine
- Evaluate: evaluate, judge, justify, critique, recommend
- Create: design, develop, generate, construct, create

## Requirements
1. Generate **minimum 5** learning objectives
2. Bloom's level distribution required:
   - Remember/Understand: 1-2
   - Apply/Analyze: 2-3
   - Evaluate/Create: 1-2
3. Format: "Learners will be able to [verb] [content]"

## Output Format (JSON Array)
```json
[
  {{"id": "LO-001", "level": "Understand", "statement": "Learners will be able to explain the organization's vision and mission.", "bloom_verb": "explain", "measurable": true}},
  {{"id": "LO-002", "level": "Apply", "statement": "Learners will be able to apply work processes to real situations.", "bloom_verb": "apply", "measurable": true}},
  {{"id": "LO-003", "level": "Analyze", "statement": "Learners will be able to analyze the strengths and weaknesses of team collaboration methods.", "bloom_verb": "analyze", "measurable": true}},
  {{"id": "LO-004", "level": "Apply", "statement": "Learners will be able to use internal systems to perform work tasks.", "bloom_verb": "use", "measurable": true}},
  {{"id": "LO-005", "level": "Evaluate", "statement": "Learners will be able to evaluate the effectiveness of communication methods.", "bloom_verb": "evaluate", "measurable": true}}
]
```

Output JSON array only."""

        response = llm.invoke(prompt)
        return parse_json_response(response.content)
    except Exception:
        return _fallback_design_objectives(learning_goals, target_audience, difficulty)


def _fallback_design_objectives(
    learning_goals: list[str],
    target_audience: str,
    difficulty: Optional[str] = None,
) -> list[dict]:
    """Fallback function when LLM fails"""
    objectives = []
    levels = ["Understand", "Apply", "Analyze", "Apply", "Evaluate"]

    for i, goal in enumerate(learning_goals[:5]):
        level = levels[i % len(levels)]
        verbs = BLOOM_VERBS.get(level, ["understand"])
        verb = verbs[0]
        statement = f"Learners will be able to {verb} {goal}."

        objectives.append({
            "id": f"LO-{i+1:03d}",
            "level": level,
            "statement": statement,
            "bloom_verb": verb,
            "measurable": True,
        })

    while len(objectives) < 5:
        idx = len(objectives)
        level = levels[idx % len(levels)]
        verbs = BLOOM_VERBS.get(level, ["understand"])
        objectives.append({
            "id": f"LO-{idx+1:03d}",
            "level": level,
            "statement": f"Learners will be able to {verbs[0]} the learning content.",
            "bloom_verb": verbs[0],
            "measurable": True,
        })

    return objectives


@tool
def design_assessment(
    objectives: list[dict],
    duration: str,
    learning_environment: str,
) -> dict:
    """
    Develop assessment plan.

    Args:
        objectives: Learning objectives list
        duration: Learning duration
        learning_environment: Learning environment

    Returns:
        Assessment plan (2+ diagnostic, 2+ formative, 2+ summative)
    """
    try:
        llm = get_llm()

        prompt = f"""You are an instructional design expert with 20 years of experience. Develop an assessment plan aligned with learning objectives.

## Input Information
- Learning Objectives: {json.dumps(objectives, ensure_ascii=False)}
- Learning Duration: {duration}
- Learning Environment: {learning_environment}

## Assessment Types
1. Diagnostic assessment: Identify prior knowledge before learning
2. Formative assessment: Check understanding during learning
3. Summative assessment: Evaluate final achievement after learning

## Requirements
- **Minimum 2** assessment methods per type
- Select appropriate methods for the learning environment (online/in-person)
- Specific methods connected to learning objectives

## Output Format (JSON)
```json
{{
  "diagnostic": [
    "Prior knowledge quiz (10 items)",
    "Self-assessment readiness checklist"
  ],
  "formative": [
    "Module-based comprehension quiz",
    "Mid-practice checkpoint",
    "Peer feedback"
  ],
  "summative": [
    "Comprehensive assessment (20 items)",
    "Practical assignment evaluation"
  ]
}}
```

Output JSON only."""

        response = llm.invoke(prompt)
        return parse_json_response(response.content)
    except Exception:
        return _fallback_design_assessment(objectives, duration, learning_environment)


def _fallback_design_assessment(
    objectives: list[dict],
    duration: str,
    learning_environment: str,
) -> dict:
    """Fallback function when LLM fails"""
    env_lower = learning_environment.lower()
    is_online = "online" in env_lower

    diagnostic = ["Prior knowledge quiz (10 items)", "Learning readiness self-checklist"]
    formative = ["Module-based comprehension quiz"]
    summative = ["Comprehensive assessment (20 items)"]

    if is_online:
        formative.append("Online practice assignment submission")
        summative.append("Video presentation evaluation")
    else:
        formative.append("Group discussion participation")
        summative.append("Practical demonstration evaluation")

    return {
        "diagnostic": diagnostic,
        "formative": formative,
        "summative": summative,
    }


GAGNE_EVENTS = [
    "Gain attention",
    "Inform learners of objectives",
    "Stimulate recall of prior learning",
    "Present content",
    "Provide learning guidance",
    "Elicit performance",
    "Provide feedback",
    "Assess performance",
    "Enhance retention and transfer",
]


@tool
def design_strategy(
    main_topics: list[str],
    target_audience: str,
    duration: str,
    learning_environment: str,
) -> dict:
    """
    Design instructional strategy based on Gagné's 9 Events.

    Args:
        main_topics: Main learning topics
        target_audience: Target learners
        duration: Learning duration
        learning_environment: Learning environment

    Returns:
        Instructional strategy (all 9 Events included, 3+ methods)
    """
    try:
        llm = get_llm()

        prompt = f"""You are an instructional design expert with 20 years of experience. Design an instructional strategy based on Gagné's 9 Events.

## Input Information
- Main Learning Topics: {json.dumps(main_topics, ensure_ascii=False)}
- Target Audience: {target_audience}
- Learning Duration: {duration}
- Learning Environment: {learning_environment}

## Gagné's 9 Events (All Required!)
1. Gain attention
2. Inform learners of objectives
3. Stimulate recall of prior learning
4. Present content
5. Provide learning guidance
6. Elicit performance
7. Provide feedback
8. Assess performance
9. Enhance retention and transfer

## Requirements
- Include **all 9 Events**
- Specify concrete activities, time, and resources for each Event
- Include **minimum 3** instructional methods

## Output Format (JSON)
```json
{{
  "model": "Gagné's 9 Events",
  "sequence": [
    {{"event": "Gain attention", "activity": "Present engaging real-world case video", "duration": "5 min", "resources": ["Case video", "Projector"]}},
    {{"event": "Inform learners of objectives", "activity": "Present today's learning content and goals", "duration": "3 min", "resources": ["Slides"]}},
    {{"event": "Stimulate recall of prior learning", "activity": "Questions and review of related prior knowledge", "duration": "5 min", "resources": ["Whiteboard"]}},
    {{"event": "Present content", "activity": "Core concept explanation and demonstration", "duration": "20 min", "resources": ["Presentation", "Demo materials"]}},
    {{"event": "Provide learning guidance", "activity": "Facilitate understanding through examples and non-examples", "duration": "10 min", "resources": ["Worksheet"]}},
    {{"event": "Elicit performance", "activity": "Individual/group practice activities", "duration": "15 min", "resources": ["Practice materials"]}},
    {{"event": "Provide feedback", "activity": "Immediate feedback on practice results", "duration": "5 min", "resources": ["Rubric"]}},
    {{"event": "Assess performance", "activity": "Quiz to confirm learning objective achievement", "duration": "5 min", "resources": ["Assessment sheet"]}},
    {{"event": "Enhance retention and transfer", "activity": "Discuss real-world application methods", "duration": "7 min", "resources": ["Discussion guide"]}}
  ],
  "methods": ["Lecture", "Demonstration", "Practice", "Discussion", "Case study"]
}}
```

Output JSON only."""

        response = llm.invoke(prompt)
        return parse_json_response(response.content)
    except Exception:
        return _fallback_design_strategy(main_topics, target_audience, duration, learning_environment)


def _fallback_design_strategy(
    main_topics: list[str],
    target_audience: str,
    duration: str,
    learning_environment: str,
) -> dict:
    """Fallback function when LLM fails"""
    topic = main_topics[0] if main_topics else "learning topic"
    env_lower = learning_environment.lower()
    is_online = "online" in env_lower

    sequence = [
        {"event": "Gain attention", "activity": f"Present engaging case related to {topic}", "duration": "5 min", "resources": ["Case materials"]},
        {"event": "Inform learners of objectives", "activity": "Present today's learning objectives", "duration": "3 min", "resources": ["Slides"]},
        {"event": "Stimulate recall of prior learning", "activity": "Review related prior knowledge", "duration": "5 min", "resources": ["Questions"]},
        {"event": "Present content", "activity": f"Explain core concepts of {topic}", "duration": "20 min", "resources": ["Presentation"]},
        {"event": "Provide learning guidance", "activity": "Facilitate understanding through examples and demonstrations", "duration": "10 min", "resources": ["Example materials"]},
        {"event": "Elicit performance", "activity": "Individual/group practice activities", "duration": "15 min", "resources": ["Practice materials"]},
        {"event": "Provide feedback", "activity": "Feedback on practice results", "duration": "5 min", "resources": ["Feedback form"]},
        {"event": "Assess performance", "activity": "Confirm learning objective achievement", "duration": "5 min", "resources": ["Assessment sheet"]},
        {"event": "Enhance retention and transfer", "activity": "Discuss real-world application methods", "duration": "7 min", "resources": ["Discussion guide"]},
    ]

    methods = ["Lecture", "Demonstration", "Practice"]
    if is_online:
        methods.extend(["Screen sharing", "Online discussion"])
    else:
        methods.extend(["Group discussion", "Role play"])

    return {
        "model": "Gagné's 9 Events",
        "sequence": sequence,
        "methods": methods,
    }
