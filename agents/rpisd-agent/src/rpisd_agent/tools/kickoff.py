"""
Project Kickoff Stage Tools

RPISD's first stage: Project Kickoff Meeting
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
def kickoff_meeting(
    project_title: str,
    learning_goals: List[str],
    target_audience: str,
    duration: Optional[str] = None,
    stakeholders: Optional[List[str]] = None,
    constraints: Optional[List[str]] = None,
) -> dict:
    """
    Conduct project kickoff meeting.

    As the first step of rapid prototyping, defines project scope and
    formalizes stakeholder roles.

    Args:
        project_title: Project title
        learning_goals: List of learning goals
        target_audience: Target learners
        duration: Training duration (optional)
        stakeholders: List of stakeholders (optional)
        constraints: Constraints (optional)

    Returns:
        Project kickoff meeting results (scope, roles, timeline, success criteria)
    """
    try:
        llm = get_llm()

        prompt = f"""You are an instructional designer specializing in rapid prototyping. Conduct a project kickoff meeting to define project scope and roles.

## Input Information
- Project Title: {project_title}
- Learning Goals: {json.dumps(learning_goals, ensure_ascii=False)}
- Target Audience: {target_audience}
- Training Duration: {duration or "TBD"}
- Stakeholders: {json.dumps(stakeholders, ensure_ascii=False) if stakeholders else "TBD"}
- Constraints: {json.dumps(constraints, ensure_ascii=False) if constraints else "None"}

## Kickoff Meeting Deliverables (Required)
1. **scope**: Project scope definition (objective, in/out of scope, key deliverables)
2. **stakeholder_roles**: Roles and responsibilities by stakeholder
3. **timeline**: High-level schedule (analysis, design, development, evaluation)
4. **success_criteria**: Success criteria (minimum 3)
5. **constraints**: Constraints and risks
6. **communication_plan**: Communication plan

## Output Format (JSON)
```json
{{
  "project_title": "{project_title}",
  "scope": {{
    "objective": "Main purpose of the project",
    "in_scope": ["In-scope item 1", "In-scope item 2"],
    "out_of_scope": ["Out-of-scope item 1"],
    "deliverables": ["Deliverable 1", "Deliverable 2", "Deliverable 3"]
  }},
  "stakeholder_roles": {{
    "sponsor": "Project approval and budget support",
    "sme": "Content expertise review and feedback",
    "learner_rep": "Learner perspective feedback",
    "designer": "Instructional design and prototype development"
  }},
  "timeline": {{
    "analysis": "Week 1",
    "design_prototype": "Weeks 2-3 (iterative)",
    "development": "Week 4",
    "implementation": "Week 5"
  }},
  "success_criteria": [
    "Learner satisfaction 4.0/5.0 or higher",
    "Learning goal achievement rate 80% or higher",
    "Schedule adherence rate 90% or higher"
  ],
  "constraints": [
    "Budget constraints",
    "Schedule constraints"
  ],
  "communication_plan": {{
    "weekly_meeting": "Every Monday at 10:00 AM",
    "feedback_channel": "Slack #project-channel",
    "review_cycle": "Upon each prototype completion"
  }}
}}
```

Output JSON only."""

        response = llm.invoke(prompt)
        return parse_json_response(response.content)
    except Exception:
        return _fallback_kickoff_meeting(
            project_title, learning_goals, target_audience,
            duration, stakeholders, constraints
        )


def _fallback_kickoff_meeting(
    project_title: str,
    learning_goals: List[str],
    target_audience: str,
    duration: Optional[str] = None,
    stakeholders: Optional[List[str]] = None,
    constraints: Optional[List[str]] = None,
) -> dict:
    """Fallback function when LLM fails"""
    return {
        "project_title": project_title,
        "scope": {
            "objective": f"Develop {project_title} training program for {target_audience}",
            "in_scope": learning_goals[:3] if learning_goals else ["Achieve learning goals"],
            "out_of_scope": ["Advanced courses", "Certification program"],
            "deliverables": [
                "Learner analysis report",
                "Prototypes (v1-v3)",
                "Final training materials",
                "Facilitator guide",
            ],
        },
        "stakeholder_roles": {
            "sponsor": "Project approval and budget support",
            "sme": "Content expertise review and feedback",
            "learner_rep": "Learner perspective feedback",
            "designer": "Instructional design and prototype development",
        },
        "timeline": {
            "analysis": "Week 1",
            "design_prototype": "Weeks 2-3 (iterative)",
            "development": "Week 4",
            "implementation": "Week 5",
        },
        "success_criteria": [
            "Learner satisfaction 4.0/5.0 or higher",
            "Learning goal achievement rate 80% or higher",
            "Prototype quality criteria met (0.8 or higher)",
        ],
        "constraints": constraints or ["Budget constraints", "Schedule constraints"],
        "communication_plan": {
            "weekly_meeting": "Weekly regular meetings",
            "feedback_channel": "Email and messenger",
            "review_cycle": "Upon each prototype completion",
        },
    }
