"""
Development Phase Tools (1 tool)

Fifth phase of RPISD: Final Program Development
- develop_final_program: Final development based on prototype
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
def develop_final_program(
    final_prototype: dict,
    aggregated_feedback: dict,
    design_result: dict,
    project_title: Optional[str] = None,
) -> dict:
    """
    Develops the final educational program.

    Develops completed educational materials and lesson plans
    based on the validated prototype.

    Args:
        final_prototype: Final prototype
        aggregated_feedback: Aggregated feedback
        design_result: Instructional design result
        project_title: Project title

    Returns:
        Final development result (lesson_plan, modules, materials, quiz_items)
    """
    try:
        llm = get_llm()

        prompt = f"""You are an instructional designer specializing in rapid prototyping. Please develop the final program based on the validated prototype.

## Project Title
{project_title or "Educational Program"}

## Final Prototype
{json.dumps(final_prototype, ensure_ascii=False, indent=2)}

## Aggregated Feedback (Items to Reflect)
{json.dumps(aggregated_feedback, ensure_ascii=False, indent=2)}

## Instructional Design
{json.dumps(design_result, ensure_ascii=False, indent=2)}

## Development Items (Final Deliverables)
1. **lesson_plan**: Completed lesson plan
2. **modules**: Completed learning modules (minimum 3)
3. **materials**: Completed learning materials (minimum 5)
4. **quiz_items**: Assessment items (minimum 10)
5. **slide_contents**: Detailed slide contents

## Output Format (JSON)
```json
{{
  "lesson_plan": {{
    "title": "{project_title or 'Educational Program'}",
    "total_duration": "2 hours",
    "overview": "Program overview",
    "objectives": ["Can explain core concepts", "Can apply to real situations"]
  }},
  "modules": [
    {{
      "title": "Module 1: Basic Concepts",
      "duration": "40 minutes",
      "objectives": ["Can explain core concepts", "Can understand basic principles"],
      "activities": [
        {{"sequence": 1, "type": "Lecture", "duration": "15 minutes", "description": "Core concept explanation", "resources": ["PPT"]}},
        {{"sequence": 2, "type": "Practice", "duration": "20 minutes", "description": "Hands-on activity", "resources": ["Worksheet"]}},
        {{"sequence": 3, "type": "Discussion", "duration": "5 minutes", "description": "Q&A", "resources": []}}
      ]
    }}
  ],
  "materials": [
    {{
      "type": "PPT",
      "title": "Main Presentation",
      "description": "Core content slides",
      "slides": 25
    }},
    {{
      "type": "Worksheet",
      "title": "Practice Guide",
      "description": "Step-by-step practice instructions",
      "pages": 5
    }},
    {{
      "type": "Handout",
      "title": "Summary Material",
      "description": "Core content summary",
      "pages": 2
    }},
    {{
      "type": "Quiz",
      "title": "Learning Check Quiz",
      "description": "Comprehension assessment",
      "questions": 10
    }},
    {{
      "type": "Video",
      "title": "Demo Video",
      "description": "Practice demonstration",
      "duration": "5 minutes"
    }}
  ],
  "quiz_items": [
    {{
      "question": "Assessment item about core concepts",
      "type": "Multiple Choice",
      "options": ["A", "B", "C", "D"],
      "answer": "A",
      "explanation": "Explanation",
      "related_objective": "Can explain core concepts",
      "difficulty": "Easy"
    }}
  ],
  "slide_contents": [
    {{
      "slide_number": 1,
      "title": "Title Slide",
      "bullet_points": ["Program Title", "Date"],
      "speaker_notes": "Greeting and introduction",
      "visual_suggestion": "Title design"
    }}
  ]
}}
```

Output only JSON."""

        response = llm.invoke(prompt)
        return parse_json_response(response.content)
    except Exception:
        return _fallback_develop_final_program(
            final_prototype, design_result, project_title
        )


def _fallback_develop_final_program(
    final_prototype: dict,
    design_result: dict,
    project_title: Optional[str] = None,
) -> dict:
    """Fallback function when LLM fails"""
    title = project_title or "Educational Program"
    objectives = design_result.get("objectives", [])

    # Helper function to extract objective statements
    def get_objective_statements(obj_list):
        return [obj.get("statement", "Achieve learning objective") for obj in obj_list]

    # Create final modules based on prototype modules
    prototype_modules = final_prototype.get("modules", [])
    modules = []
    for i, pm in enumerate(prototype_modules[:3]):
        # Use objectives as-is if already in sentence form, otherwise convert
        pm_objectives = pm.get("objectives", [])
        if pm_objectives and isinstance(pm_objectives[0], str) and not pm_objectives[0].startswith("LO-"):
            module_objectives = pm_objectives
        else:
            module_objectives = get_objective_statements(objectives[i*2:(i+1)*2]) or [f"Achieve Module {i+1} learning objective"]

        modules.append({
            "title": pm.get("title", f"Module {i+1}: Learning Content"),
            "duration": pm.get("duration", "30 minutes"),
            "objectives": module_objectives,
            "activities": [
                {"sequence": 1, "type": "Lecture", "duration": "15 minutes", "description": "Core concept explanation", "resources": ["PPT"]},
                {"sequence": 2, "type": "Practice", "duration": "10 minutes", "description": "Hands-on activity", "resources": ["Worksheet"]},
                {"sequence": 3, "type": "Discussion", "duration": "5 minutes", "description": "Q&A", "resources": []},
            ],
        })

    while len(modules) < 3:
        idx = len(modules)
        module_objectives = get_objective_statements(objectives[idx*2:(idx+1)*2]) or [f"Can understand additional learning content"]
        modules.append({
            "title": f"Module {idx+1}: Additional Learning",
            "duration": "30 minutes",
            "objectives": module_objectives,
            "activities": [
                {"sequence": 1, "type": "Lecture", "duration": "15 minutes", "description": "Content explanation", "resources": ["PPT"]},
                {"sequence": 2, "type": "Practice", "duration": "15 minutes", "description": "Practice", "resources": ["Worksheet"]},
            ],
        })

    # Learning materials
    materials = [
        {"type": "PPT", "title": f"{title} Presentation", "description": "Main learning material", "slides": 25},
        {"type": "Worksheet", "title": "Practice Guide", "description": "Practice instructions", "pages": 5},
        {"type": "Handout", "title": "Summary Material", "description": "Core content summary", "pages": 2},
        {"type": "Quiz", "title": "Learning Check Quiz", "description": "Assessment items", "questions": 10},
        {"type": "Video", "title": "Demo Video", "description": "Demonstration video", "duration": "5 minutes"},
    ]

    # Quiz items (reference objective statements instead of IDs)
    quiz_items = []
    for i in range(10):
        related_objective = objectives[i % len(objectives)].get("statement", "Achieve learning objective") if objectives else "Achieve learning objective"
        difficulty = ["Easy", "Medium", "Hard"][i % 3]
        quiz_items.append({
            "question": f"Assessment item {i+1} about {related_objective[:20]}",
            "type": "Multiple Choice" if i < 7 else "Short Answer",
            "options": ["A", "B", "C", "D"] if i < 7 else None,
            "answer": "A" if i < 7 else "Example answer",
            "explanation": f"Explanation for item {i+1}",
            "related_objective": related_objective,
            "difficulty": difficulty,
        })

    # Slide contents
    objective_statements = get_objective_statements(objectives[:3]) or ["Learning objective 1", "Learning objective 2", "Learning objective 3"]
    slide_contents = [
        {"slide_number": 1, "title": title, "bullet_points": ["Educational Program", "Date"], "speaker_notes": "Greeting", "visual_suggestion": "Title"},
        {"slide_number": 2, "title": "Learning Objectives", "bullet_points": [s[:30] for s in objective_statements], "speaker_notes": "Objective introduction", "visual_suggestion": "Icons"},
        {"slide_number": 3, "title": "Table of Contents", "bullet_points": [m["title"] for m in modules], "speaker_notes": "Structure introduction", "visual_suggestion": "Table of contents"},
    ]

    return {
        "lesson_plan": {
            "title": title,
            "total_duration": "2 hours",
            "overview": f"{title} educational program",
            "objectives": get_objective_statements(objectives[:5]) or ["Develop core competencies", "Improve practical application skills"],
        },
        "modules": modules,
        "materials": materials,
        "quiz_items": quiz_items,
        "slide_contents": slide_contents,
    }
