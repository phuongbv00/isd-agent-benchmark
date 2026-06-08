"""
Objective & Assessment Tools (Steps 4-5)

Dick & Carey Model's Steps 4-5:
4. Performance Objectives
5. Assessment Instruments
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


# ========== Step 4: Performance Objectives ==========
@tool
def write_performance_objectives(
    instructional_goal: str,
    sub_skills: List[dict],
    target_audience: str,
) -> dict:
    """
    Write performance objectives. (Dick & Carey Step 4)

    Write specific and measurable performance objectives in ABCD format.
    - A (Audience): Target learners
    - B (Behavior): Observable behavior
    - C (Condition): Performance conditions
    - D (Degree): Achievement criteria

    Args:
        instructional_goal: Instructional goal
        sub_skills: Sub-skills list
        target_audience: Target learners

    Returns:
        Performance objectives result (terminal_objective, enabling_objectives)
    """
    try:
        llm = get_llm()

        prompt = f"""You are an expert in the Dick & Carey model. Write Performance Objectives based on the following information.

## Input Information
- Instructional Goal: {instructional_goal}
- Sub-skills: {json.dumps(sub_skills, ensure_ascii=False)}
- Target Audience: {target_audience}

## Dick & Carey's Performance Objective Principles (ABCD)
- A (Audience): Specify target learners
- B (Behavior): Use observable and measurable action verbs
- C (Condition): Conditions under which performance occurs
- D (Degree): Acceptable performance level/criteria

## Bloom's Taxonomy Levels
- Remember, Understand, Apply, Analyze, Evaluate, Create

## Required Elements
1. terminal_objective: One terminal performance objective (ABCD format)
2. enabling_objectives: Enabling performance objectives **minimum 5** (corresponding to each sub-skill)

## Output Format (JSON)
**Important**:
- Use objective_name field instead of id field.
- Use actual sub-skill description text in sub_skill_id.

```json
{{
  "terminal_objective": {{
    "objective_name": "Comprehensive performance objective",
    "audience": "{target_audience}",
    "behavior": "Can independently perform given tasks",
    "condition": "When provided with relevant materials and tools",
    "degree": "90% or higher accuracy, within time limit",
    "statement": "{target_audience} can independently perform given tasks when provided with relevant materials and tools. (90% or higher accuracy, within time limit)",
    "sub_skill_id": "Comprehensive task performance",
    "bloom_level": "Apply"
  }},
  "enabling_objectives": [
    {{
      "objective_name": "Core concept understanding objective",
      "audience": "{target_audience}",
      "behavior": "Can define and explain core concepts",
      "condition": "Without textbooks or reference materials",
      "degree": "At least 5 key concepts accurately",
      "statement": "{target_audience} can accurately define and explain at least 5 core concepts without textbooks or reference materials.",
      "sub_skill_id": "Can explain core concepts",
      "bloom_level": "Understand"
    }},
    {{
      "objective_name": "Case analysis objective",
      "audience": "{target_audience}",
      "behavior": "Can analyze given cases",
      "condition": "When provided with an analysis framework",
      "degree": "Identify at least 3 key elements",
      "statement": "{target_audience} can identify at least 3 key elements from given cases when provided with an analysis framework.",
      "sub_skill_id": "Can analyze related cases",
      "bloom_level": "Analyze"
    }},
    {{
      "objective_name": "Problem diagnosis objective",
      "audience": "{target_audience}",
      "behavior": "Can diagnose problem situations",
      "condition": "In real or simulated situations",
      "degree": "Accurately identify at least 2 problems",
      "statement": "{target_audience} can accurately diagnose at least 2 problems in real or simulated situations.",
      "sub_skill_id": "Can diagnose problem situations",
      "bloom_level": "Analyze"
    }},
    {{
      "objective_name": "Solution design objective",
      "audience": "{target_audience}",
      "behavior": "Can design and apply solutions",
      "condition": "Under limited resource conditions",
      "degree": "At least 1 feasible solution",
      "statement": "{target_audience} can design and apply at least one feasible solution under limited resource conditions.",
      "sub_skill_id": "Can design solutions",
      "bloom_level": "Apply"
    }},
    {{
      "objective_name": "Result evaluation objective",
      "audience": "{target_audience}",
      "behavior": "Can evaluate results and suggest improvements",
      "condition": "When provided with evaluation criteria",
      "degree": "Identify at least 2 improvements",
      "statement": "{target_audience} can evaluate results and suggest at least 2 improvements when provided with evaluation criteria.",
      "sub_skill_id": "Can evaluate results and make improvements",
      "bloom_level": "Evaluate"
    }}
  ]
}}
```

Output JSON only."""

        response = llm.invoke(prompt)
        return parse_json_response(response.content)
    except Exception:
        return _fallback_write_performance_objectives(instructional_goal, sub_skills, target_audience)


def _fallback_write_performance_objectives(
    instructional_goal: str,
    sub_skills: List[dict],
    target_audience: str,
) -> dict:
    """Fallback function when LLM fails"""
    enabling = []
    bloom_levels = ["Understand", "Analyze", "Analyze", "Apply", "Evaluate"]
    objective_names = ["Core concept understanding objective", "Case analysis objective", "Problem diagnosis objective", "Solution design objective", "Result evaluation objective"]

    for i, skill in enumerate(sub_skills[:5]):
        description = skill.get("description", f"Sub-skill {i+1}")
        bloom = bloom_levels[i] if i < len(bloom_levels) else "Apply"
        obj_name = objective_names[i] if i < len(objective_names) else f"{description} objective"

        enabling.append({
            "objective_name": obj_name,
            "audience": target_audience,
            "behavior": description,
            "condition": "Under appropriate conditions",
            "degree": "80% or higher accuracy",
            "statement": f"{target_audience} can {description} under appropriate conditions. (80% or higher accuracy)",
            "sub_skill_id": description,
            "bloom_level": bloom,
        })

    # Ensure minimum 5
    while len(enabling) < 5:
        i = len(enabling)
        additional_behavior = f"Can apply learning content ({i+1})"
        enabling.append({
            "objective_name": f"Learning application objective {i+1}",
            "audience": target_audience,
            "behavior": additional_behavior,
            "condition": "Under appropriate conditions",
            "degree": "80% or higher accuracy",
            "statement": f"{target_audience} can apply learning content under appropriate conditions. (80% or higher accuracy)",
            "sub_skill_id": additional_behavior,
            "bloom_level": "Apply",
        })

    return {
        "terminal_objective": {
            "objective_name": "Comprehensive performance objective",
            "audience": target_audience,
            "behavior": "Can comprehensively perform learning content",
            "condition": "When provided with necessary resources",
            "degree": "90% or higher accuracy",
            "statement": f"{target_audience} can comprehensively perform learning content when provided with necessary resources. (90% or higher accuracy)",
            "sub_skill_id": "Comprehensive task performance",
            "bloom_level": "Apply",
        },
        "enabling_objectives": enabling,
    }


# ========== Step 5: Assessment Instruments ==========
@tool
def develop_assessment_instruments(
    performance_objectives: dict,
    learning_environment: str,
    duration: str,
) -> dict:
    """
    Develop assessment instruments. (Dick & Carey Step 5)

    Develop entry test, practice tests, and post-test aligned with performance objectives.
    Ensures objective-assessment alignment.

    Args:
        performance_objectives: Performance objectives (terminal_objective, enabling_objectives)
        learning_environment: Learning environment
        duration: Learning duration

    Returns:
        Assessment instruments result (entry_test, practice_tests, post_test, alignment_matrix)
    """
    try:
        llm = get_llm()

        prompt = f"""You are an expert in the Dick & Carey model. Develop Assessment Instruments aligned with performance objectives.

## Input Information
- Performance Objectives: {json.dumps(performance_objectives, ensure_ascii=False)}
- Learning Environment: {learning_environment}
- Duration: {duration}

## Dick & Carey's Assessment Instrument Development Principles
1. Objective-Assessment Alignment: Each assessment item measures specific performance objective
2. Criterion-Referenced Assessment: Judge whether learner performance meets objective criteria
3. Assessment Types:
   - Entry Test: Confirm entry behaviors
   - Practice Test: Practice and feedback during learning
   - Post-test: Confirm objective achievement

## Required Elements
1. entry_test: Entry test items **minimum 3**
2. practice_tests: Practice test items **minimum 3**
3. post_test: Post-test items **minimum 5**
4. alignment_matrix: Objective-assessment alignment matrix

## Output Format (JSON)
**Important**:
- Use actual performance objective text (behavior) in objective_id.
- Use text instead of IDs for both keys and values in alignment_matrix.
- Use assessment_name field instead of id field for each assessment item.

```json
{{
  "entry_test": [
    {{
      "assessment_name": "Basic concept multiple choice assessment",
      "objective_id": "Basic knowledge verification",
      "type": "multiple_choice",
      "question": "Which of the following corresponds to [basic concept]?",
      "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
      "answer": "Option 1",
      "rubric": "1 point for correct answer"
    }},
    {{
      "assessment_name": "Basic terminology short answer assessment",
      "objective_id": "Basic knowledge verification",
      "type": "short_answer",
      "question": "Briefly explain the meaning of [basic terminology].",
      "options": [],
      "answer": "Scored based on inclusion of key keywords",
      "rubric": "Full marks for including 2 or more key keywords"
    }},
    {{
      "assessment_name": "Basic concept true/false assessment",
      "objective_id": "Basic knowledge verification",
      "type": "true_false",
      "question": "[Basic concept statement] is a correct explanation.",
      "options": ["True", "False"],
      "answer": "True",
      "rubric": "1 point for correct answer"
    }}
  ],
  "practice_tests": [
    {{
      "assessment_name": "Core concept essay practice",
      "objective_id": "Can explain core concepts",
      "type": "essay",
      "question": "Explain the learned concept in your own words.",
      "options": [],
      "answer": "Inclusion of key concepts",
      "rubric": "Full marks for accurately describing 3 or more core concepts"
    }},
    {{
      "assessment_name": "Case analysis practice",
      "objective_id": "Can analyze related cases",
      "type": "case_analysis",
      "question": "Analyze the key elements in the presented case.",
      "options": [],
      "answer": "Key element identification",
      "rubric": "Full marks for accurately identifying 2 or more key elements"
    }},
    {{
      "assessment_name": "Problem diagnosis practice",
      "objective_id": "Can diagnose problem situations",
      "type": "problem_solving",
      "question": "Identify problems in the given situation.",
      "options": [],
      "answer": "Problem identification",
      "rubric": "Full marks for accurately identifying 2 or more problems"
    }}
  ],
  "post_test": [
    {{
      "assessment_name": "Core concept multiple choice assessment",
      "objective_id": "Can explain core concepts",
      "type": "multiple_choice",
      "question": "Which of the following best explains [core concept]?",
      "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
      "answer": "Option 2",
      "rubric": "2 points for correct answer"
    }},
    {{
      "assessment_name": "Case analysis essay assessment",
      "objective_id": "Can analyze related cases",
      "type": "essay",
      "question": "Analyze the presented case and identify key elements.",
      "options": [],
      "answer": "Analysis results",
      "rubric": "Full marks for 3 or more key elements with logical explanation"
    }},
    {{
      "assessment_name": "Problem diagnosis assessment",
      "objective_id": "Can diagnose problem situations",
      "type": "problem_solving",
      "question": "Diagnose the given problem situation and analyze causes.",
      "options": [],
      "answer": "Diagnosis results",
      "rubric": "Full marks for accurately analyzing 2 or more problem causes"
    }},
    {{
      "assessment_name": "Solution design performance assessment",
      "objective_id": "Can design and apply solutions",
      "type": "performance_assessment",
      "question": "Perform the actual task and submit results.",
      "options": [],
      "answer": "Task results",
      "rubric": "Evaluation by completeness, accuracy, and applicability"
    }},
    {{
      "assessment_name": "Comprehensive application assessment",
      "objective_id": "Can comprehensively perform learning content",
      "type": "comprehensive_assessment",
      "question": "Apply learning content comprehensively to a real situation.",
      "options": [],
      "answer": "Comprehensive application",
      "rubric": "Rubric evaluation based on goal achievement criteria"
    }}
  ],
  "alignment_matrix": {{
    "Can explain core concepts": ["Basic concept multiple choice assessment", "Core concept essay practice", "Core concept multiple choice assessment"],
    "Can analyze related cases": ["Case analysis practice", "Case analysis essay assessment"],
    "Can diagnose problem situations": ["Problem diagnosis practice", "Problem diagnosis assessment"],
    "Can design and apply solutions": ["Solution design performance assessment"],
    "Can comprehensively perform learning content": ["Comprehensive application assessment"]
  }}
}}
```

Output JSON only."""

        response = llm.invoke(prompt)
        return parse_json_response(response.content)
    except Exception:
        return _fallback_develop_assessment_instruments(performance_objectives, learning_environment, duration)


def _fallback_develop_assessment_instruments(
    performance_objectives: dict,
    learning_environment: str,
    duration: str,
) -> dict:
    """Fallback function when LLM fails"""
    enabling = performance_objectives.get("enabling_objectives", [])

    # entry_test: Use assessment_name instead of ID
    entry_test = [
        {"assessment_name": "Basic concept multiple choice assessment", "objective_id": "Basic knowledge verification", "type": "multiple_choice", "question": "Basic concept verification item", "options": ["A", "B", "C", "D"], "answer": "A", "rubric": "1 point for correct"},
        {"assessment_name": "Basic terminology short answer assessment", "objective_id": "Basic knowledge verification", "type": "short_answer", "question": "Basic terminology explanation", "options": [], "answer": "Include key keywords", "rubric": "Full marks for 2 or more keywords"},
        {"assessment_name": "Basic concept true/false assessment", "objective_id": "Basic knowledge verification", "type": "true_false", "question": "Basic concept true/false item", "options": ["True", "False"], "answer": "True", "rubric": "1 point for correct"},
    ]

    practice_tests = []
    practice_names = []  # For alignment_matrix values
    for i, obj in enumerate(enabling[:3]):
        behavior = obj.get("behavior", f"Learning content {i+1}")
        assessment_name = f"{behavior} essay practice"
        practice_names.append((behavior, assessment_name))
        practice_tests.append({
            "assessment_name": assessment_name,
            "objective_id": behavior,
            "type": "essay",
            "question": f"Practice item related to {behavior}",
            "options": [],
            "answer": "Model answer",
            "rubric": "Full marks for including key content",
        })

    while len(practice_tests) < 3:
        i = len(practice_tests)
        default_behavior = f"Can apply learning content ({i+1})"
        assessment_name = f"{default_behavior} practice"
        practice_names.append((default_behavior, assessment_name))
        practice_tests.append({
            "assessment_name": assessment_name,
            "objective_id": default_behavior,
            "type": "essay",
            "question": f"Practice item {i+1}",
            "options": [],
            "answer": "Model answer",
            "rubric": "Full marks for including key content",
        })

    post_test = []
    post_names = []  # For alignment_matrix values
    for i, obj in enumerate(enabling[:4]):
        behavior = obj.get("behavior", f"Learning content {i+1}")
        test_type = "essay" if i % 2 == 0 else "multiple_choice"
        assessment_name = f"{behavior} {test_type} assessment"
        post_names.append((behavior, assessment_name))
        post_test.append({
            "assessment_name": assessment_name,
            "objective_id": behavior,
            "type": test_type,
            "question": f"{behavior} assessment item",
            "options": ["A", "B", "C", "D"] if i % 2 == 1 else [],
            "answer": "A" if i % 2 == 1 else "Model answer",
            "rubric": "Full marks for accurate performance",
        })

    # Ensure minimum 4 (before adding comprehensive item)
    while len(post_test) < 4:
        i = len(post_test)
        default_behavior = f"Can apply learning content ({i+1})"
        test_type = "essay" if i % 2 == 0 else "multiple_choice"
        assessment_name = f"{default_behavior} {test_type} assessment"
        post_names.append((default_behavior, assessment_name))
        post_test.append({
            "assessment_name": assessment_name,
            "objective_id": default_behavior,
            "type": test_type,
            "question": f"Learning content assessment item {i+1}",
            "options": ["A", "B", "C", "D"] if i % 2 == 1 else [],
            "answer": "A" if i % 2 == 1 else "Model answer",
            "rubric": "Full marks for accurate performance",
        })

    # Add final comprehensive item
    terminal = performance_objectives.get("terminal_objective", {})
    terminal_behavior = terminal.get("behavior", "Can comprehensively perform learning content")
    terminal_assessment_name = "Comprehensive application assessment"
    post_test.append({
        "assessment_name": terminal_assessment_name,
        "objective_id": terminal_behavior,
        "type": "comprehensive_assessment",
        "question": f"{terminal_behavior} comprehensive assessment",
        "options": [],
        "answer": "Comprehensive assessment criteria",
        "rubric": "Rubric evaluation based on goal achievement criteria",
    })

    # alignment_matrix: Use text for both keys and values
    alignment = {}
    # Match practice_names and post_names by behavior
    behavior_to_assessments = {}
    for behavior, name in practice_names:
        if behavior not in behavior_to_assessments:
            behavior_to_assessments[behavior] = []
        behavior_to_assessments[behavior].append(name)
    for behavior, name in post_names:
        if behavior not in behavior_to_assessments:
            behavior_to_assessments[behavior] = []
        behavior_to_assessments[behavior].append(name)

    alignment = behavior_to_assessments
    alignment[terminal_behavior] = [terminal_assessment_name]

    return {
        "entry_test": entry_test,
        "practice_tests": practice_tests,
        "post_test": post_test,
        "alignment_matrix": alignment,
    }
