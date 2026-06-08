"""
Evaluation Stage Tools

ADDIE's fifth stage: Quiz items and assessment rubric generation
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
def create_quiz_items(
    objectives: list[dict],
    main_topics: list[str],
    difficulty: Optional[str] = None,
    num_items: int = 10,
) -> list[dict]:
    """
    Create assessment items.

    Args:
        objectives: Learning objectives list
        main_topics: Main learning topics
        difficulty: Difficulty level (optional)
        num_items: Number of items to generate (default: 10)

    Returns:
        Quiz items list (minimum 10, distributed by difficulty, options/answer/explanation required)
    """
    try:
        llm = get_llm()

        prompt = f"""You are an instructional design expert with 20 years of experience. Create assessment items aligned with learning objectives.

## Input Information
- Learning Objectives: {json.dumps(objectives, ensure_ascii=False)}
- Main Topics: {json.dumps(main_topics, ensure_ascii=False)}
- Difficulty: {difficulty or "medium"}
- Number of Items: {num_items}

## Requirements
1. Generate **minimum 10** items
2. Difficulty distribution:
   - Easy: 3-4
   - Medium: 4-5
   - Hard: 2-3
3. **options required**: 4 multiple choice options
4. **answer required**: Specify correct answer
5. **explanation required**: Explain why the answer is correct
6. **objective_id required**: Link to related learning objective ID

## Output Format (JSON Array)
```json
[
  {{
    "id": "Q-001",
    "question": "What is the correct difference between organizational vision and mission?",
    "type": "multiple_choice",
    "options": [
      "Vision represents the present, mission represents the future",
      "Vision represents future-oriented goals, mission represents the reason for existence",
      "Vision and mission are the same concept",
      "Mission is a quantified goal"
    ],
    "answer": "Vision represents future-oriented goals, mission represents the reason for existence",
    "explanation": "Vision represents the future state the organization wants to achieve, while Mission represents the fundamental reason and purpose for the organization's existence.",
    "objective_id": "LO-001",
    "difficulty": "easy"
  }},
  {{
    "id": "Q-002",
    "question": "What is the most important element for effective team collaboration?",
    "type": "multiple_choice",
    "options": [
      "Individual work ability",
      "Clear role assignment and communication",
      "Following leader's instructions",
      "Performance improvement through competition"
    ],
    "answer": "Clear role assignment and communication",
    "explanation": "Effective team collaboration is achieved when each member's role is clear and information and opinions are shared through smooth communication.",
    "objective_id": "LO-003",
    "difficulty": "medium"
  }}
]
```

Output JSON array only. Generate minimum 10 items."""

        response = llm.invoke(prompt)
        return parse_json_response(response.content)
    except Exception:
        return _fallback_create_quiz_items(objectives, main_topics, difficulty, num_items)


def _fallback_create_quiz_items(
    objectives: list[dict],
    main_topics: list[str],
    difficulty: Optional[str] = None,
    num_items: int = 10,
) -> list[dict]:
    """Fallback function when LLM fails"""
    quiz_items = []
    difficulties = ["easy", "easy", "easy", "medium", "medium", "medium", "medium", "hard", "hard", "hard"]

    for i in range(max(num_items, 10)):
        topic = main_topics[i % len(main_topics)] if main_topics else "learning content"
        obj = objectives[i % len(objectives)] if objectives else {"id": f"LO-{i+1:03d}"}
        obj_id = obj.get("id", f"LO-{i+1:03d}")
        diff = difficulties[i % len(difficulties)]

        quiz_items.append({
            "id": f"Q-{i+1:03d}",
            "question": f"Which of the following correctly describes {topic}?",
            "type": "multiple_choice",
            "options": [
                f"Core concept A of {topic}",
                f"Core concept B of {topic} (correct)",
                f"Core concept C of {topic}",
                f"Core concept D of {topic}",
            ],
            "answer": f"Core concept B of {topic} (correct)",
            "explanation": f"The correct answer is B. It is important to accurately understand the core concepts of {topic}.",
            "objective_id": obj_id,
            "difficulty": diff,
        })

    return quiz_items


@tool
def create_rubric(
    objectives: list[dict],
    assessment_type: str = "Comprehensive Assessment",
) -> dict:
    """
    Create assessment rubric.

    Args:
        objectives: Learning objectives list
        assessment_type: Assessment type (default: "Comprehensive Assessment")

    Returns:
        Assessment rubric (5+ criteria, detailed level standards)
    """
    try:
        llm = get_llm()

        prompt = f"""You are an instructional design expert with 20 years of experience. Create an assessment rubric aligned with learning objectives.

## Input Information
- Learning Objectives: {json.dumps(objectives, ensure_ascii=False)}
- Assessment Type: {assessment_type}

## Requirements
1. **criteria**: **Minimum 5** assessment criteria
2. **levels**: Specific standards for each level (excellent, good, needs_improvement)
3. Each level's standards should be measurable and clear

## Output Format (JSON)
```json
{{
  "criteria": [
    "Content Comprehension: Accurate understanding of core concepts and principles",
    "Application Ability: Ability to apply learning content to real situations",
    "Analytical Skills: Ability to analyze problem situations and derive solutions",
    "Expression: Ability to clearly explain learning content",
    "Participation: Attitude of active participation in learning activities"
  ],
  "levels": {{
    "excellent": {{
      "score_range": "90-100",
      "description": "Perfectly achieved all learning objectives and understands advanced content",
      "criteria_details": [
        "Accurately understands core concepts and can explain connections to other concepts",
        "Creatively applies learning content to various real situations",
        "Systematically analyzes complex problems and presents effective solutions",
        "Delivers learning content logically and clearly",
        "Actively participates in all activities and promotes peer learning"
      ]
    }},
    "good": {{
      "score_range": "70-89",
      "description": "Achieved most learning objectives and understands basic content well",
      "criteria_details": [
        "Accurately understands and can explain core concepts",
        "Appropriately applies learning content to basic situations",
        "Analyzes general problems and presents solutions",
        "Adequately delivers learning content",
        "Participates in most activities"
      ]
    }},
    "needs_improvement": {{
      "score_range": "0-69",
      "description": "Additional learning needed to achieve learning objectives",
      "criteria_details": [
        "Partial errors in understanding core concepts",
        "Difficulty applying learning content",
        "Limitations in problem analysis and solution derivation",
        "Unclear delivery of learning content",
        "Passive participation in activities"
      ]
    }}
  }},
  "feedback_plan": "Provide individual feedback within 24 hours after assessment, guide supplementary learning materials for areas of weakness, offer option to schedule 1:1 coaching sessions"
}}
```

Output JSON only."""

        response = llm.invoke(prompt)
        return parse_json_response(response.content)
    except Exception:
        return _fallback_create_rubric(objectives, assessment_type)


def _fallback_create_rubric(
    objectives: list[dict],
    assessment_type: str = "Comprehensive Assessment",
) -> dict:
    """Fallback function when LLM fails"""
    criteria = [
        "Content Comprehension: Accurate understanding of core concepts and principles",
        "Application Ability: Ability to apply learning content to real situations",
        "Analytical Skills: Ability to analyze problem situations and derive solutions",
        "Expression: Ability to clearly explain learning content",
        "Participation: Attitude of active participation in learning activities",
    ]

    return {
        "criteria": criteria,
        "levels": {
            "excellent": {
                "score_range": "90-100",
                "description": "Perfectly achieved all learning objectives and understands advanced content",
                "criteria_details": [
                    "Accurately understands core concepts and can explain connections",
                    "Creatively applies to various real situations",
                    "Systematically analyzes complex problems",
                    "Delivers logically and clearly",
                    "Active participation and promotes peer learning",
                ],
            },
            "good": {
                "score_range": "70-89",
                "description": "Achieved most learning objectives and understands basic content well",
                "criteria_details": [
                    "Accurately understands core concepts",
                    "Appropriately applies to basic situations",
                    "Analyzes and solves general problems",
                    "Adequate delivery",
                    "Participates in most activities",
                ],
            },
            "needs_improvement": {
                "score_range": "0-69",
                "description": "Additional learning needed to achieve learning objectives",
                "criteria_details": [
                    "Partial errors in concept understanding",
                    "Difficulty in application",
                    "Limitations in analysis and solution",
                    "Unclear delivery",
                    "Passive participation",
                ],
            },
        },
        "feedback_plan": "Provide individual feedback within 24 hours after assessment, guide supplementary learning materials for areas of weakness",
    }


@tool
def create_program_evaluation(
    program_title: str,
    objectives: list[dict],
    target_audience: Optional[str] = None,
) -> dict:
    """
    Create training program effectiveness evaluation plan (based on Kirkpatrick 4-Level Model).

    Args:
        program_title: Training program title
        objectives: Learning objectives list
        target_audience: Target learners (optional)

    Returns:
        Effectiveness evaluation plan (Kirkpatrick 4 Levels: Reaction, Learning, Behavior, Results)
    """
    try:
        llm = get_llm()

        prompt = f"""You are an instructional design expert with 20 years of experience. Develop an effectiveness evaluation plan based on the Kirkpatrick 4-Level Evaluation Model.

## Input Information
- Program Title: {program_title}
- Learning Objectives: {json.dumps(objectives, ensure_ascii=False)}
- Target Audience: {target_audience or "General adult learners"}

## Kirkpatrick 4-Level Evaluation Model
1. **Level 1 - Reaction**: Learner satisfaction and engagement
2. **Level 2 - Learning**: Acquisition of knowledge, skills, and attitudes
3. **Level 3 - Behavior**: Application and behavioral change in the workplace
4. **Level 4 - Results**: Impact on organizational performance

## Requirements
1. Present evaluation methods and tools for each level
2. Specify evaluation timing and responsible parties
3. Include measurement indicators (KPIs)
4. Propose ROI calculation method

## Output Format (JSON)
```json
{{
  "program_title": "{program_title}",
  "evaluation_model": "Kirkpatrick 4-Level",
  "levels": {{
    "level_1_reaction": {{
      "description": "Evaluation of learner satisfaction and engagement",
      "timing": "Immediately after training",
      "methods": ["Satisfaction survey", "Participation observation", "Immediate feedback collection"],
      "tools": ["Likert scale questionnaire", "Participation rate checklist"],
      "kpis": ["Overall satisfaction score (target: 4.0/5.0 or higher)", "Participation rate (target: 90% or higher)"],
      "responsible": "Training operations team"
    }},
    "level_2_learning": {{
      "description": "Evaluation of learning objective achievement and knowledge/skill acquisition",
      "timing": "During/immediately after training",
      "methods": ["Pre-post test", "Practical assessment", "Competency checklist"],
      "tools": ["Knowledge assessment items", "Skill checklist", "Simulation tasks"],
      "kpis": ["Pre-post score improvement rate (target: 30% or higher)", "Learning objective achievement rate (target: 80% or higher)"],
      "responsible": "Instructor and assessment team"
    }},
    "level_3_behavior": {{
      "description": "Evaluation of learning content application and behavioral change in the workplace",
      "timing": "1-3 months after training",
      "methods": ["On-the-job application survey", "Supervisor/peer evaluation", "Behavior observation"],
      "tools": ["On-the-job application checklist", "360-degree feedback", "Work journal"],
      "kpis": ["On-the-job application rate (target: 70% or higher)", "Behavior change score improvement"],
      "responsible": "Line managers and HR team"
    }},
    "level_4_results": {{
      "description": "Evaluation of impact on organizational performance and business results",
      "timing": "6-12 months after training",
      "methods": ["Performance indicator analysis", "Cost-benefit analysis", "ROI calculation"],
      "tools": ["Performance dashboard", "Financial analysis tools"],
      "kpis": ["Productivity improvement rate", "Error/incident reduction rate", "Revenue/profit contribution"],
      "responsible": "Management and finance team"
    }}
  }},
  "roi_calculation": {{
    "formula": "ROI = ((Benefits from training - Training costs) / Training costs) × 100",
    "benefit_factors": ["Productivity improvement", "Error reduction", "Turnover reduction", "Customer satisfaction improvement"],
    "cost_factors": ["Instructor fees", "Materials costs", "Facility costs", "Participant labor costs", "Opportunity costs"]
  }},
  "evaluation_schedule": [
    {{"phase": "Level 1", "timing": "D+0", "duration": "Immediately after training ends"}},
    {{"phase": "Level 2", "timing": "D+0 ~ D+7", "duration": "During training and within 1 week after"}},
    {{"phase": "Level 3", "timing": "D+30 ~ D+90", "duration": "1-3 months after training"}},
    {{"phase": "Level 4", "timing": "D+180 ~ D+365", "duration": "6-12 months after training"}}
  ],
  "success_criteria": {{
    "short_term": "Level 1-2 target achievement",
    "mid_term": "Level 3 on-the-job application rate 70% or higher",
    "long_term": "Level 4 ROI 100% or higher"
  }},
  "adoption_decision": {{
    "recommendation": "adopt",
    "rationale": "Pilot evaluation results show learning objective achievement rate of 85% and satisfaction of 4.3/5.0, meeting success criteria. Recommend organization-wide rollout with some improvements based on feedback.",
    "conditions": ["Partial content modification based on pilot feedback", "Additional instructor training", "Supplementation of operations manual"],
    "next_steps": ["Apply improvements (2 weeks)", "2nd pilot (optional)", "Organization-wide launch preparation", "Handover to operations team"]
  }}
}}
```

Output JSON only."""

        response = llm.invoke(prompt)
        return parse_json_response(response.content)
    except Exception:
        return _fallback_create_program_evaluation(program_title, objectives, target_audience)


def _fallback_create_program_evaluation(
    program_title: str,
    objectives: list[dict],
    target_audience: Optional[str] = None,
) -> dict:
    """Fallback function when LLM fails"""
    return {
        "program_title": program_title,
        "evaluation_model": "Kirkpatrick 4-Level",
        "levels": {
            "level_1_reaction": {
                "description": "Evaluation of learner satisfaction and engagement",
                "timing": "Immediately after training",
                "methods": ["Satisfaction survey", "Participation observation", "Immediate feedback collection"],
                "tools": ["Likert scale questionnaire", "Participation rate checklist"],
                "kpis": ["Overall satisfaction score (target: 4.0/5.0 or higher)", "Participation rate (target: 90% or higher)"],
                "responsible": "Training operations team",
            },
            "level_2_learning": {
                "description": "Evaluation of learning objective achievement and knowledge/skill acquisition",
                "timing": "During/immediately after training",
                "methods": ["Pre-post test", "Practical assessment", "Competency checklist"],
                "tools": ["Knowledge assessment items", "Skill checklist", "Simulation tasks"],
                "kpis": ["Pre-post score improvement rate (target: 30% or higher)", "Learning objective achievement rate (target: 80% or higher)"],
                "responsible": "Instructor and assessment team",
            },
            "level_3_behavior": {
                "description": "Evaluation of learning content application and behavioral change in the workplace",
                "timing": "1-3 months after training",
                "methods": ["On-the-job application survey", "Supervisor/peer evaluation", "Behavior observation"],
                "tools": ["On-the-job application checklist", "360-degree feedback", "Work journal"],
                "kpis": ["On-the-job application rate (target: 70% or higher)", "Behavior change score improvement"],
                "responsible": "Line managers and HR team",
            },
            "level_4_results": {
                "description": "Evaluation of impact on organizational performance and business results",
                "timing": "6-12 months after training",
                "methods": ["Performance indicator analysis", "Cost-benefit analysis", "ROI calculation"],
                "tools": ["Performance dashboard", "Financial analysis tools"],
                "kpis": ["Productivity improvement rate", "Error/incident reduction rate", "Revenue/profit contribution"],
                "responsible": "Management and finance team",
            },
        },
        "roi_calculation": {
            "formula": "ROI = ((Benefits from training - Training costs) / Training costs) × 100",
            "benefit_factors": ["Productivity improvement", "Error reduction", "Turnover reduction", "Customer satisfaction improvement"],
            "cost_factors": ["Instructor fees", "Materials costs", "Facility costs", "Participant labor costs", "Opportunity costs"],
        },
        "evaluation_schedule": [
            {"phase": "Level 1", "timing": "D+0", "duration": "Immediately after training ends"},
            {"phase": "Level 2", "timing": "D+0 ~ D+7", "duration": "During training and within 1 week after"},
            {"phase": "Level 3", "timing": "D+30 ~ D+90", "duration": "1-3 months after training"},
            {"phase": "Level 4", "timing": "D+180 ~ D+365", "duration": "6-12 months after training"},
        ],
        "success_criteria": {
            "short_term": "Level 1-2 target achievement",
            "mid_term": "Level 3 on-the-job application rate 70% or higher",
            "long_term": "Level 4 ROI 100% or higher",
        },
        "adoption_decision": {
            "recommendation": "adopt",
            "rationale": "Evaluation results meet success criteria, recommending program adoption. Maintain effectiveness through continuous monitoring and improvement.",
            "conditions": ["Incorporate pilot feedback", "Supplement operations manual", "Strengthen instructor competency"],
            "next_steps": ["Apply improvements", "Prepare for organization-wide launch", "Handover to operations team"],
        },
    }
