"""
Development Stage Tools

ADDIE's third stage: Lesson plan and learning materials development
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
def create_lesson_plan(
    objectives: list[dict],
    instructional_strategy: dict,
    duration: str,
    main_topics: list[str],
) -> dict:
    """
    Create lesson plan.

    Args:
        objectives: Learning objectives list
        instructional_strategy: Instructional strategy
        duration: Total learning duration
        main_topics: Main learning topics

    Returns:
        Lesson plan (3+ modules, 3+ activities per module)
    """
    try:
        llm = get_llm()

        prompt = f"""You are an instructional design expert with 20 years of experience. Create a systematic lesson plan based on the following information.

## Input Information
- Learning Objectives: {json.dumps(objectives, ensure_ascii=False)}
- Instructional Strategy: {json.dumps(instructional_strategy, ensure_ascii=False)}
- Total Learning Duration: {duration}
- Main Topics: {json.dumps(main_topics, ensure_ascii=False)}

## Requirements
1. **Minimum 3** modules
2. **Minimum 3** activities per module
3. Each module connected to learning objectives
4. Time allocation matches total learning duration

## Output Format (JSON)
```json
{{
  "total_duration": "{duration}",
  "modules": [
    {{
      "title": "Module 1: Understanding Organizational Culture",
      "duration": "30 min",
      "objectives": ["LO-001", "LO-002"],
      "activities": [
        {{"time": "0-5 min", "activity": "Introduction", "description": "Watch organizational culture case video", "resources": ["Video materials"]}},
        {{"time": "5-15 min", "activity": "Lecture", "description": "Explain organizational vision and mission", "resources": ["Presentation"]}},
        {{"time": "15-25 min", "activity": "Discussion", "description": "Connect organizational values with personal values", "resources": ["Discussion guide"]}},
        {{"time": "25-30 min", "activity": "Wrap-up", "description": "Key content summary and Q&A", "resources": []}}
      ]
    }},
    {{
      "title": "Module 2: Work Processes",
      "duration": "35 min",
      "objectives": ["LO-003", "LO-004"],
      "activities": [
        {{"time": "0-5 min", "activity": "Introduction", "description": "Explain importance of work processes", "resources": ["Slides"]}},
        {{"time": "5-20 min", "activity": "Demonstration", "description": "Demonstrate core work procedures", "resources": ["System demo"]}},
        {{"time": "20-30 min", "activity": "Practice", "description": "System usage practice", "resources": ["Practice environment"]}},
        {{"time": "30-35 min", "activity": "Feedback", "description": "Check practice results and provide feedback", "resources": []}}
      ]
    }},
    {{
      "title": "Module 3: Team Collaboration",
      "duration": "25 min",
      "objectives": ["LO-005"],
      "activities": [
        {{"time": "0-5 min", "activity": "Introduction", "description": "Introduce collaboration tools", "resources": ["Tool list"]}},
        {{"time": "5-15 min", "activity": "Practice", "description": "Collaboration tool usage practice", "resources": ["Collaboration platform"]}},
        {{"time": "15-25 min", "activity": "Role play", "description": "Team collaboration scenario role play", "resources": ["Scenario cards"]}}
      ]
    }}
  ]
}}
```

Output JSON only."""

        response = llm.invoke(prompt)
        return parse_json_response(response.content)
    except Exception:
        return _fallback_create_lesson_plan(objectives, instructional_strategy, duration, main_topics)


def _fallback_create_lesson_plan(
    objectives: list[dict],
    instructional_strategy: dict,
    duration: str,
    main_topics: list[str],
) -> dict:
    """Fallback function when LLM fails"""
    modules = []

    for i, topic in enumerate(main_topics[:3]):
        obj_ids = [f"LO-{i*2+1:03d}", f"LO-{i*2+2:03d}"] if i < len(objectives) // 2 else [f"LO-{i+1:03d}"]

        modules.append({
            "title": f"Module {i+1}: {topic}",
            "duration": "30 min",
            "objectives": obj_ids,
            "activities": [
                {"time": "0-5 min", "activity": "Introduction", "description": f"Overview of {topic}", "resources": ["Slides"]},
                {"time": "5-15 min", "activity": "Lecture", "description": f"Explain core content of {topic}", "resources": ["Presentation"]},
                {"time": "15-25 min", "activity": "Practice", "description": f"Practice related to {topic}", "resources": ["Practice materials"]},
                {"time": "25-30 min", "activity": "Wrap-up", "description": "Key content summary", "resources": []},
            ],
        })

    while len(modules) < 3:
        idx = len(modules) + 1
        modules.append({
            "title": f"Module {idx}: Additional Learning",
            "duration": "20 min",
            "objectives": [f"LO-{idx:03d}"],
            "activities": [
                {"time": "0-5 min", "activity": "Introduction", "description": "Learning content overview", "resources": []},
                {"time": "5-15 min", "activity": "Learning", "description": "Core content learning", "resources": ["Materials"]},
                {"time": "15-20 min", "activity": "Wrap-up", "description": "Summary and Q&A", "resources": []},
            ],
        })

    return {
        "total_duration": duration,
        "modules": modules,
    }


@tool
def create_materials(
    lesson_plan: dict,
    learning_environment: str,
    target_audience: str,
) -> list[dict]:
    """
    Create learning materials.

    Args:
        lesson_plan: Lesson plan
        learning_environment: Learning environment
        target_audience: Target learners

    Returns:
        Learning materials list (minimum 5, includes slide_contents)
    """
    try:
        llm = get_llm()

        prompt = f"""You are an instructional design expert with 20 years of experience. Create learning materials based on the lesson plan.

## Input Information
- Lesson Plan: {json.dumps(lesson_plan, ensure_ascii=False)}
- Learning Environment: {learning_environment}
- Target Audience: {target_audience}

## Requirements
1. **Minimum 5** learning materials
2. PPT/Presentation must include **slide_contents**
3. Include various types (PPT, video, quiz, worksheet, handout)
4. slides, pages values must be numbers (no null)

## Output Format (JSON Array)
```json
[
  {{
    "type": "PPT",
    "title": "Organizational Culture Understanding Slides",
    "description": "Materials explaining organizational vision, mission, and core values",
    "slides": 10,
    "slide_contents": [
      {{"slide_number": 1, "title": "Training Introduction", "bullet_points": ["Welcome greeting", "Learning objectives", "Schedule overview"], "speaker_notes": "Welcome participants and explain training overview"}},
      {{"slide_number": 2, "title": "Organizational Vision", "bullet_points": ["Vision definition", "Meaning of vision", "Our vision"], "speaker_notes": "Explain organizational vision in detail"}},
      {{"slide_number": 3, "title": "Organizational Mission", "bullet_points": ["Mission definition", "Difference between mission and vision", "Our mission"], "speaker_notes": "Explain relationship between mission and vision"}}
    ],
    "storyboard": [
      {{"frame_number": 1, "screen_title": "Introduction Screen", "visual_description": "Company logo and training title displayed", "audio_narration": "Welcome to the organizational culture training.", "interaction": "Click start button", "notes": "Background music fade in"}},
      {{"frame_number": 2, "screen_title": "Learning Objectives", "visual_description": "Learning objectives list animation", "audio_narration": "Let's review today's learning content and objectives.", "interaction": "Auto-advance", "notes": "Display objectives sequentially"}}
    ]
  }},
  {{
    "type": "Video",
    "title": "Organization Introduction Video",
    "description": "Video introducing company history and culture",
    "duration": "5 min",
    "storyboard": [
      {{"frame_number": 1, "screen_title": "Opening", "visual_description": "Drone shot of company exterior", "audio_narration": "We introduce our company's history and culture.", "interaction": "None", "notes": "Background music"}}
    ]
  }},
  {{
    "type": "Worksheet",
    "title": "Work Process Practice Sheet",
    "description": "Worksheet for practicing core work procedures",
    "pages": 3
  }},
  {{
    "type": "Quiz",
    "title": "Module-based Comprehension Quiz",
    "description": "Quiz to check understanding after each module",
    "questions": 15
  }},
  {{
    "type": "Handout",
    "title": "Key Content Summary Material",
    "description": "Distribution material summarizing training core content",
    "pages": 5
  }}
]
```

Output JSON array only."""

        response = llm.invoke(prompt)
        return parse_json_response(response.content)
    except Exception:
        return _fallback_create_materials(lesson_plan, learning_environment, target_audience)


def _fallback_create_materials(
    lesson_plan: dict,
    learning_environment: str,
    target_audience: str,
) -> list[dict]:
    """Fallback function when LLM fails"""
    modules = lesson_plan.get("modules", [])
    env_lower = learning_environment.lower()
    is_online = "online" in env_lower

    materials = []

    # PPT materials (includes slide_contents)
    slide_contents = [
        {"slide_number": 1, "title": "Training Introduction", "bullet_points": ["Welcome greeting", "Learning objectives", "Schedule overview"], "speaker_notes": "Welcome participants and explain training overview"},
        {"slide_number": 2, "title": "Learning Objectives", "bullet_points": ["Today's learning objectives", "Expected outcomes", "Assessment methods"], "speaker_notes": "Clearly communicate learning objectives"},
    ]

    for i, module in enumerate(modules):
        title = module.get("title", f"Module {i+1}")
        slide_contents.append({
            "slide_number": len(slide_contents) + 1,
            "title": title,
            "bullet_points": ["Core concepts", "Key content", "Practice guide"],
            "speaker_notes": f"Explain {title} content",
        })

    slide_contents.append({
        "slide_number": len(slide_contents) + 1,
        "title": "Summary and Q&A",
        "bullet_points": ["Key content summary", "Q&A", "Next steps"],
        "speaker_notes": "Wrap up training and answer questions",
    })

    # Create storyboard
    storyboard = [
        {
            "frame_number": 1,
            "screen_title": "Introduction Screen",
            "visual_description": "Training title and logo displayed",
            "audio_narration": "Welcome to the training.",
            "interaction": "Click start button",
            "notes": "Background music fade in",
        },
        {
            "frame_number": 2,
            "screen_title": "Learning Objectives",
            "visual_description": "Learning objectives list animation",
            "audio_narration": "Let's review today's learning content.",
            "interaction": "Auto-advance",
            "notes": "Display objectives sequentially",
        },
    ]

    materials.append({
        "type": "PPT",
        "title": "Training Presentation",
        "description": "Slides for complete training session",
        "slides": len(slide_contents),
        "slide_contents": slide_contents,
        "storyboard": storyboard,
    })

    # Video materials
    if is_online:
        materials.append({
            "type": "Video",
            "title": "Pre-learning Video",
            "description": "Video content for online pre-learning",
            "duration": "10 min",
        })
    else:
        materials.append({
            "type": "Video",
            "title": "Introduction Case Video",
            "description": "Real case video related to learning topic",
            "duration": "5 min",
        })

    # Worksheet
    materials.append({
        "type": "Worksheet",
        "title": "Practice Worksheet",
        "description": "Worksheet for module-based practice activities",
        "pages": 4,
    })

    # Quiz
    materials.append({
        "type": "Quiz",
        "title": "Comprehension Quiz",
        "description": "Quiz to check learning content for each module",
        "questions": 15,
    })

    # Handout
    materials.append({
        "type": "Handout",
        "title": "Key Content Summary",
        "description": "Distribution material summarizing training core content",
        "pages": 3,
    })

    return materials
