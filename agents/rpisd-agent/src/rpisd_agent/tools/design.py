"""
Design Phase Tools (3 tools)

Third phase of RPISD: Design and Prototype Development
- design_instruction: Instructional design
- develop_prototype: Prototype development (iterative)
- analyze_task_detailed: Detailed task analysis
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
def design_instruction(
    learning_goals: List[str],
    learner_characteristics: dict,
    duration: Optional[str] = None,
    learning_environment: Optional[str] = None,
) -> dict:
    """
    Performs instructional design.

    Designs instructional strategies and learning objectives
    based on learning goals and learner characteristics.

    Args:
        learning_goals: List of learning goals
        learner_characteristics: Learner characteristics analysis results
        duration: Training duration
        learning_environment: Learning environment

    Returns:
        Instructional design results (objectives, strategy, sequence)
    """
    try:
        llm = get_llm()

        prompt = f"""You are an instructional designer specializing in rapid prototyping. Please perform instructional design for rapid prototype development.

## Input Information
- Learning Goals: {json.dumps(learning_goals, ensure_ascii=False)}
- Learner Characteristics: {json.dumps(learner_characteristics, ensure_ascii=False)}
- Training Duration: {duration or "Undetermined"}
- Learning Environment: {learning_environment or "Undetermined"}

## Design Items (Required)
1. **objectives**: Learning objectives (based on Bloom's Taxonomy, minimum 5)
2. **strategy**: Instructional strategy (based on Gagné's 9 Events)
3. **sequence**: Learning sequence (9 Events)
4. **methods**: Instructional methods

## Output Format (JSON)
```json
{{
  "objectives": [
    {{
      "level": "Understand",
      "statement": "Can explain core concepts",
      "bloom_verb": "explain",
      "measurable": true
    }},
    {{
      "level": "Apply",
      "statement": "Can apply learned content to real situations",
      "bloom_verb": "apply",
      "measurable": true
    }}
  ],
  "strategy": {{
    "model": "Gagné's 9 Events",
    "approach": "Hands-on focused, immediate feedback",
    "media": ["Presentation", "Practice materials", "Quiz"]
  }},
  "sequence": [
    {{"event": "Gain Attention", "activity": "Present relevant cases or questions", "duration": "5 min"}},
    {{"event": "Inform Objectives", "activity": "Clearly state learning objectives", "duration": "3 min"}},
    {{"event": "Stimulate Recall", "activity": "Review related prior knowledge", "duration": "5 min"}},
    {{"event": "Present Content", "activity": "Explain new content", "duration": "15 min"}},
    {{"event": "Provide Guidance", "activity": "Emphasize key points", "duration": "5 min"}},
    {{"event": "Elicit Performance", "activity": "Practice and exercises", "duration": "20 min"}},
    {{"event": "Provide Feedback", "activity": "Immediate feedback", "duration": "5 min"}},
    {{"event": "Assess Performance", "activity": "Quiz or assignment", "duration": "10 min"}},
    {{"event": "Enhance Retention and Transfer", "activity": "Discuss application methods", "duration": "7 min"}}
  ],
  "methods": ["Lecture", "Demonstration", "Practice", "Discussion", "Feedback"]
}}
```

Output only JSON."""

        response = llm.invoke(prompt)
        return parse_json_response(response.content)
    except Exception:
        return _fallback_design_instruction(learning_goals, duration)


def _fallback_design_instruction(
    learning_goals: List[str],
    duration: Optional[str] = None,
) -> dict:
    """Fallback function when LLM fails"""
    objectives = []
    bloom_levels = ["Understand", "Apply", "Analyze", "Evaluate", "Create"]
    bloom_verbs = ["explain", "apply", "analyze", "evaluate", "develop"]

    for i, goal in enumerate(learning_goals[:5]):
        objectives.append({
            "level": bloom_levels[i % len(bloom_levels)],
            "statement": f"Can perform {goal}",
            "bloom_verb": bloom_verbs[i % len(bloom_verbs)],
            "measurable": True,
        })

    while len(objectives) < 5:
        idx = len(objectives)
        objectives.append({
            "level": bloom_levels[idx % len(bloom_levels)],
            "statement": f"Can {bloom_verbs[idx % len(bloom_verbs)]} the learning content",
            "bloom_verb": bloom_verbs[idx % len(bloom_verbs)],
            "measurable": True,
        })

    return {
        "objectives": objectives,
        "strategy": {
            "model": "Gagné's 9 Events",
            "approach": "Hands-on focused, immediate feedback",
            "media": ["Presentation", "Practice materials", "Quiz"],
        },
        "sequence": [
            {"event": "Gain Attention", "activity": "Present relevant cases", "duration": "5 min"},
            {"event": "Inform Objectives", "activity": "State learning objectives", "duration": "3 min"},
            {"event": "Stimulate Recall", "activity": "Review related knowledge", "duration": "5 min"},
            {"event": "Present Content", "activity": "Explain new content", "duration": "15 min"},
            {"event": "Provide Guidance", "activity": "Emphasize key points", "duration": "5 min"},
            {"event": "Elicit Performance", "activity": "Practice and exercises", "duration": "20 min"},
            {"event": "Provide Feedback", "activity": "Immediate feedback", "duration": "5 min"},
            {"event": "Assess Performance", "activity": "Quiz", "duration": "10 min"},
            {"event": "Enhance Retention and Transfer", "activity": "Discuss application methods", "duration": "7 min"},
        ],
        "methods": ["Lecture", "Demonstration", "Practice", "Discussion", "Feedback"],
    }


@tool
def develop_prototype(
    design_result: dict,
    prototype_version: int = 1,
    previous_feedback: Optional[List[dict]] = None,
    focus_areas: Optional[List[str]] = None,
) -> dict:
    """
    Develops a prototype.

    This is the core step of rapid prototyping, where prototypes are
    quickly created to collect feedback. Iteratively improved by version.

    Args:
        design_result: Instructional design results
        prototype_version: Prototype version number
        previous_feedback: Previous version feedback (for iterations)
        focus_areas: Areas to focus on for improvement (for iterations)

    Returns:
        Prototype (modules, materials, sample_content)
    """
    try:
        llm = get_llm()

        feedback_context = ""
        if previous_feedback and prototype_version > 1:
            feedback_context = f"""
## Previous Feedback (v{prototype_version - 1})
{json.dumps(previous_feedback, ensure_ascii=False, indent=2)}

## Focus Areas for Improvement
{json.dumps(focus_areas, ensure_ascii=False) if focus_areas else "Overall improvement"}

**Please create an improved prototype reflecting the previous feedback.**
"""

        prompt = f"""You are an instructional designer specializing in rapid prototyping. Please develop prototype v{prototype_version}.

## Instructional Design Results
{json.dumps(design_result, ensure_ascii=False, indent=2)}
{feedback_context}
## Prototype Components (Required)
1. **version**: Version number
2. **modules**: Learning modules (minimum 3)
3. **materials**: Learning materials specifications (minimum 3)
4. **sample_content**: Sample content (at least 3 slides)

## Output Format (JSON)
```json
{{
  "version": {prototype_version},
  "modules": [
    {{
      "title": "Module 1: Basic Concepts",
      "duration": "30 min",
      "objectives": ["Can explain core concepts", "Can apply learned content to real situations"],
      "activities": [
        {{"type": "Lecture", "duration": "15 min", "description": "Explain core concepts"}},
        {{"type": "Practice", "duration": "15 min", "description": "Concept application exercises"}}
      ]
    }}
  ],
  "materials": [
    {{
      "type": "PPT",
      "title": "Learning Material Slides",
      "slides": 15,
      "description": "Core content presentation"
    }},
    {{
      "type": "Worksheet",
      "title": "Practice Worksheet",
      "pages": 3,
      "description": "Practice activity guide"
    }}
  ],
  "sample_content": [
    {{
      "slide_number": 1,
      "title": "Learning Objectives",
      "bullet_points": ["Objective 1", "Objective 2", "Objective 3"],
      "speaker_notes": "Clearly present the learning objectives."
    }},
    {{
      "slide_number": 2,
      "title": "Core Concepts",
      "bullet_points": ["Concept 1", "Concept 2"],
      "speaker_notes": "Explain the core concepts."
    }},
    {{
      "slide_number": 3,
      "title": "Practice Guide",
      "bullet_points": ["Practice 1", "Practice 2"],
      "speaker_notes": "Guide the practice methods."
    }}
  ],
  "improvement_notes": "v{prototype_version} major changes"
}}
```

Output only JSON."""

        response = llm.invoke(prompt)
        return parse_json_response(response.content)
    except Exception:
        return _fallback_develop_prototype(design_result, prototype_version, previous_feedback)


def _fallback_develop_prototype(
    design_result: dict,
    prototype_version: int = 1,
    previous_feedback: Optional[List[dict]] = None,
) -> dict:
    """Fallback function when LLM fails"""
    objectives = design_result.get("objectives", [])

    # Extract objective statements (use actual statement instead of ID)
    def get_objective_statements(obj_list):
        return [obj.get("statement", "Achieve learning objective") for obj in obj_list]

    modules = [
        {
            "title": "Module 1: Basic Concepts",
            "duration": "30 min",
            "objectives": get_objective_statements(objectives[:2]) or ["Can explain core concepts"],
            "activities": [
                {"type": "Lecture", "duration": "15 min", "description": "Explain core concepts"},
                {"type": "Practice", "duration": "15 min", "description": "Concept application exercises"},
            ],
        },
        {
            "title": "Module 2: Advanced Learning",
            "duration": "40 min",
            "objectives": get_objective_statements(objectives[2:4]) or ["Can understand advanced content"],
            "activities": [
                {"type": "Demonstration", "duration": "20 min", "description": "Demonstrate real application cases"},
                {"type": "Discussion", "duration": "20 min", "description": "Discuss application methods"},
            ],
        },
        {
            "title": "Module 3: Practice and Assessment",
            "duration": "30 min",
            "objectives": get_objective_statements(objectives[4:5]) or ["Can comprehensively apply learning content"],
            "activities": [
                {"type": "Practice", "duration": "20 min", "description": "Comprehensive practice"},
                {"type": "Assessment", "duration": "10 min", "description": "Quiz"},
            ],
        },
    ]

    improvement_notes = f"Prototype v{prototype_version}"
    if previous_feedback and prototype_version > 1:
        improvement_notes = f"Improved to v{prototype_version} reflecting feedback"

    return {
        "version": prototype_version,
        "modules": modules,
        "materials": [
            {"type": "PPT", "title": "Learning Material Slides", "slides": 15, "description": "Core content presentation"},
            {"type": "Worksheet", "title": "Practice Worksheet", "pages": 3, "description": "Practice activity guide"},
            {"type": "Quiz", "title": "Learning Verification Quiz", "questions": 10, "description": "Comprehension check"},
        ],
        "sample_content": [
            {"slide_number": 1, "title": "Learning Objectives", "bullet_points": ["Objective 1", "Objective 2", "Objective 3"], "speaker_notes": "Present learning objectives"},
            {"slide_number": 2, "title": "Core Concepts", "bullet_points": ["Concept 1", "Concept 2"], "speaker_notes": "Explain core concepts"},
            {"slide_number": 3, "title": "Practice Guide", "bullet_points": ["Practice 1", "Practice 2"], "speaker_notes": "Guide practice methods"},
        ],
        "improvement_notes": improvement_notes,
    }


@tool
def analyze_task_detailed(
    prototype: dict,
    initial_task_analysis: dict,
    feedback: Optional[dict] = None,
) -> dict:
    """
    Performs detailed task analysis.

    This is an in-depth task analysis performed after prototype development,
    refining the structure and sequence of actual learning tasks.

    Args:
        prototype: Prototype results
        initial_task_analysis: Initial task analysis results
        feedback: Usability evaluation feedback (optional)

    Returns:
        Detailed task analysis results (refined_topics, task_sequence, dependencies)
    """
    try:
        llm = get_llm()

        prompt = f"""You are an instructional designer specializing in rapid prototyping. Please perform detailed task analysis based on the prototype.

## Prototype
{json.dumps(prototype, ensure_ascii=False, indent=2)}

## Initial Task Analysis
{json.dumps(initial_task_analysis, ensure_ascii=False, indent=2)}

## Feedback (if available)
{json.dumps(feedback, ensure_ascii=False, indent=2) if feedback else "None"}

## Analysis Items (Detailed Analysis)
1. **refined_topics**: Refined topic structure
2. **task_sequence**: Task execution sequence
3. **dependencies**: Dependencies between tasks
4. **difficulty_mapping**: Difficulty mapping

## Output Format (JSON)
```json
{{
  "refined_topics": [
    {{
      "title": "Understanding Basic Concepts",
      "subtopics": ["Core terminology definitions", "Basic principles learning"],
      "importance": "high",
      "estimated_time": "30 min"
    }}
  ],
  "task_sequence": [
    {{"order": 1, "topic": "Understanding Basic Concepts", "description": "Basic concept learning"}},
    {{"order": 2, "topic": "Advanced Learning", "description": "Principle understanding"}}
  ],
  "dependencies": [
    {{"from_topic": "Understanding Basic Concepts", "to_topic": "Advanced Learning", "type": "prerequisite"}}
  ],
  "difficulty_mapping": {{
    "beginner": ["Understanding Basic Concepts"],
    "intermediate": ["Advanced Learning"],
    "advanced": ["Application and Practice"]
  }}
}}
```

Output only JSON."""

        response = llm.invoke(prompt)
        return parse_json_response(response.content)
    except Exception:
        return _fallback_analyze_task_detailed(initial_task_analysis)


def _fallback_analyze_task_detailed(
    initial_task_analysis: dict,
) -> dict:
    """Fallback function when LLM fails"""
    main_topics = initial_task_analysis.get("main_topics", ["Understanding Basic Concepts", "Advanced Learning", "Application and Practice"])

    refined_topics = []
    task_sequence = []
    dependencies = []

    for i, topic in enumerate(main_topics):
        refined_topics.append({
            "title": topic,
            "subtopics": [f"{topic} basics", f"{topic} advanced"],
            "importance": "high" if i == 0 else "medium",
            "estimated_time": "30 min",
        })
        task_sequence.append({
            "order": i + 1,
            "topic": topic,
            "description": f"{topic} learning",
        })
        if i > 0:
            dependencies.append({
                "from_topic": main_topics[i - 1],
                "to_topic": topic,
                "type": "prerequisite",
            })

    return {
        "refined_topics": refined_topics,
        "task_sequence": task_sequence,
        "dependencies": dependencies,
        "difficulty_mapping": {
            "beginner": [main_topics[0]] if main_topics else [],
            "intermediate": [main_topics[1]] if len(main_topics) > 1 else [],
            "advanced": [main_topics[2]] if len(main_topics) > 2 else [],
        },
    }
