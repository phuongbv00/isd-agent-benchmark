"""
Implementation Phase Tool (1 tool)

The final phase of RPISD: Program Implementation and Maintenance
- implement_program: Program implementation and maintenance planning
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
def implement_program(
    development_result: dict,
    learning_environment: Optional[str] = None,
    target_audience: Optional[str] = None,
    project_title: Optional[str] = None,
) -> dict:
    """
    Establishes the program implementation and maintenance plan.

    Includes methods for executing the completed program and
    a continuous maintenance plan.

    Args:
        development_result: Development results
        learning_environment: Learning environment
        target_audience: Target learners
        project_title: Project title

    Returns:
        Implementation plan (delivery_method, guides, technical_requirements, maintenance_plan)
    """
    try:
        llm = get_llm()

        prompt = f"""You are an instructional designer specializing in rapid prototyping. Please establish a program implementation and maintenance plan.

## Project Title
{project_title or "Training Program"}

## Development Results
{json.dumps(development_result, ensure_ascii=False, indent=2)}

## Learning Environment
{learning_environment or "Blended (Face-to-face/Online)"}

## Target Audience
{target_audience or "General learners"}

## Implementation Plan Items (Required)
1. **delivery_method**: Delivery method
2. **facilitator_guide**: Facilitator guide (at least 200 characters)
3. **learner_guide**: Learner guide (at least 200 characters)
4. **operator_guide**: Operator guide (at least 200 characters)
5. **technical_requirements**: Technical requirements (minimum 3 items)
6. **maintenance_plan**: Maintenance plan
7. **support_plan**: Support plan
8. **pilot_plan**: Pilot implementation plan
9. **orientation_plan**: Orientation plan
10. **monitoring_plan**: Operations monitoring plan

## Output Format (JSON)
```json
{{
  "delivery_method": "Blended learning (Online pre-learning + Face-to-face workshop)",
  "facilitator_guide": "This is a guide for facilitating this training program. Please check all materials and equipment before starting the training. Clearly present the learning objectives and adhere to the time allocation for each module. During hands-on activities, monitor individual learner progress and provide appropriate feedback. Allocate time for Q&A to check learner comprehension. After the training, conduct a satisfaction survey and record areas for improvement.",
  "learner_guide": "Thank you for participating in this training program. Please review the pre-training materials and complete any prerequisite learning if necessary. During the training, actively participate and feel free to ask questions at any time. In hands-on activities, apply what you have learned directly and collaborate with colleagues to enhance learning effectiveness. After the training, please review the provided materials and apply them to your work.",
  "operator_guide": "This is a guide for operating this training program. In the preparation phase, confirm LMS settings and trainee registration, and set up training material uploads and access permissions. On the training day, support attendance confirmation and participation encouragement, technical support standby, and instructor-learner communication. After the training, conduct satisfaction surveys, prepare result reports, and organize improvement points.",
  "technical_requirements": [
    "Video conferencing platform (Zoom/Teams) - for online sessions",
    "LMS access permissions - for pre-learning and material distribution",
    "Projector and screen - for face-to-face training",
    "Stable internet connection (minimum 10Mbps)"
  ],
  "maintenance_plan": {{
    "update_frequency": "Quarterly",
    "update_triggers": [
      "Reflecting learner feedback",
      "Content changes occurred",
      "Adding new case studies"
    ],
    "version_control": "v1.0 → v1.1 (Minor) → v2.0 (Major)",
    "responsible_party": "Instructional Design Team",
    "review_schedule": "Quarterly review meetings"
  }},
  "support_plan": {{
    "during_training": "Real-time Q&A, technical support staff on standby",
    "post_training": "Email inquiries, FAQ page, additional materials provided",
    "contact": "Training Support Team (training@company.com)"
  }},
  "pilot_plan": {{
    "phase": "Pilot test",
    "participants": "10-20 people (representative sample)",
    "duration": "1-2 weeks",
    "evaluation_criteria": ["Learning objective achievement rate", "Usability score", "Learner satisfaction"],
    "success_threshold": 0.8,
    "feedback_collection": ["Survey", "Interview", "Observation"]
  }},
  "orientation_plan": {{
    "pre_training": "Send advance notice email, verify LMS access, provide prerequisite learning guidance",
    "day_of_training": "Orientation session (15 minutes), learning objectives and schedule briefing",
    "materials_distribution": "Distribute textbooks and materials, provide access information"
  }},
  "monitoring_plan": {{
    "attendance_tracking": "Attendance and participation rate monitoring",
    "progress_monitoring": "Learning progress tracking and dropout management",
    "issue_resolution": "Real-time problem resolution support",
    "reporting": "Weekly/monthly operations report preparation"
  }}
}}
```

Output only JSON."""

        response = llm.invoke(prompt)
        return parse_json_response(response.content)
    except Exception:
        return _fallback_implement_program(
            development_result, learning_environment, target_audience, project_title
        )


def _fallback_implement_program(
    development_result: dict,
    learning_environment: Optional[str] = None,
    target_audience: Optional[str] = None,
    project_title: Optional[str] = None,
) -> dict:
    """Fallback function when LLM fails"""
    title = project_title or "Training Program"
    env = learning_environment or "Blended (Face-to-face/Online)"

    delivery_method = "Blended learning" if "online" in env.lower() or "blended" in env.lower() else env

    return {
        "delivery_method": delivery_method,
        "facilitator_guide": f"""This is a guide for facilitating the {title} program. Please check all materials and equipment before starting the training. Clearly present the learning objectives and adhere to the time allocation for each module. During hands-on activities, monitor individual learner progress and provide appropriate feedback. Allocate time for Q&A to check learner comprehension. After the training, conduct a satisfaction survey and record areas for improvement.""",
        "learner_guide": f"""Thank you for participating in the {title} program. Please review the pre-training materials and complete any prerequisite learning if necessary. During the training, actively participate and feel free to ask questions at any time. In hands-on activities, apply what you have learned directly and collaborate with colleagues to enhance learning effectiveness. After the training, please review the provided materials and apply them to your work.""",
        "technical_requirements": [
            "Video conferencing platform (Zoom/Teams)",
            "LMS access permissions",
            "Projector and screen (for face-to-face)",
            "Stable internet connection",
        ],
        "maintenance_plan": {
            "update_frequency": "Quarterly",
            "update_triggers": [
                "Reflecting learner feedback",
                "Content changes occurred",
                "Adding new case studies",
            ],
            "version_control": "v1.0 → v1.1 (Minor) → v2.0 (Major)",
            "responsible_party": "Instructional Design Team",
            "review_schedule": "Quarterly review",
        },
        "support_plan": {
            "during_training": "Real-time Q&A, technical support",
            "post_training": "Email inquiries, FAQ, additional materials",
            "contact": "Training Support Team",
        },
        "operator_guide": f"""This is a guide for operating the {title} program. In the preparation phase, confirm LMS settings and trainee registration, and set up training material uploads and access permissions. On the training day, support attendance confirmation and participation encouragement, technical support standby, and instructor-learner communication. After the training, conduct satisfaction surveys, prepare result reports, and organize improvement points.""",
        "pilot_plan": {
            "phase": "Pilot test",
            "participants": "10-20 people (representative sample)",
            "duration": "1-2 weeks",
            "evaluation_criteria": ["Learning objective achievement rate", "Usability score", "Learner satisfaction"],
            "success_threshold": 0.8,
            "feedback_collection": ["Survey", "Interview", "Observation"],
        },
        "orientation_plan": {
            "pre_training": "Send advance notice email, verify LMS access, provide prerequisite learning guidance",
            "day_of_training": "Orientation session (15 minutes), learning objectives and schedule briefing",
            "materials_distribution": "Distribute textbooks and materials, provide access information",
        },
        "monitoring_plan": {
            "attendance_tracking": "Attendance and participation rate monitoring",
            "progress_monitoring": "Learning progress tracking and dropout management",
            "issue_resolution": "Real-time problem resolution support",
            "reporting": "Weekly/monthly operations report preparation",
        },
    }
