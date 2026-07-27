"""
ADDIE 5단계별 통합 도구

각 도구는 해당 ADDIE 단계의 표준 스키마 섹션을 직접 반환합니다.
28개 개별 도구 대신 5개 단계별 도구로 통합하여 단순화합니다.
"""

import json
from typing import List, Optional
from langchain_core.tools import tool

from shared.llm import create_chat_model, get_default_llm_config


def get_llm():
    return create_chat_model(get_default_llm_config())


def parse_json_response(content: str) -> dict:
    """LLM 응답에서 JSON 파싱"""
    if "```json" in content:
        json_str = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        json_str = content.split("```")[1].split("```")[0]
    else:
        json_str = content
    return json.loads(json_str.strip())


# ============================================================
# 1. Analysis 단계 (소항목 1-10)
# ============================================================

ANALYSIS_PROMPT = """You are an instructional design expert. Perform the ADDIE Analysis phase for the following scenario.

## Scenario Information
- Title: {title}
- Target learners: {target_audience}
- Learning environment: {learning_environment}
- Duration: {duration}
- Prior knowledge: {prior_knowledge}
- Learning goals: {learning_goals}

## Analysis Phase Requirements (10 sub-items)

### A-1~4. Needs analysis (needs_analysis)
- problem_definition: Gap between the current state and the target state (2-3 sentences)
- gap_analysis: Gap between expected performance and actual performance (at least 3)
- performance_analysis: Judgment on whether an instructional solution is needed (2-3 sentences)
- priority_matrix: high/medium/low priority classification

### A-5. Learner analysis (learner_analysis)
- target_audience: Clearly define the target learners
- characteristics: At least 5 concrete characteristics
- prior_knowledge: Prior knowledge level (2-3 sentences)
- learning_preferences: At least 4 learning preferences
- motivation: Motivation level and reasons (2-3 sentences)
- challenges: At least 3 anticipated challenges

### A-6. Context analysis (context_analysis)
- environment: Learning environment
- duration: Total learning time
- constraints: At least 3 constraints
- resources: At least 3 available resources
- technical_requirements: At least 2 technical requirements

### A-7~10. Task analysis (task_analysis)
- main_topics: At least 3 main learning topics
- subtopics: At least 6 detailed learning subtopics
- prerequisites: At least 2 prerequisite learning requirements
- review_summary: Synthesis of the analysis results (3-4 sentences)

## Output Format (JSON)
```json
{{
  "needs_analysis": {{
    "problem_definition": "...",
    "gap_analysis": ["gap1", "gap2", "gap3"],
    "performance_analysis": "...",
    "priority_matrix": {{
      "high": ["item1", "item2"],
      "medium": ["item3"],
      "low": ["item4"]
    }}
  }},
  "learner_analysis": {{
    "target_audience": "...",
    "characteristics": ["characteristic1", "characteristic2", "characteristic3", "characteristic4", "characteristic5"],
    "prior_knowledge": "...",
    "learning_preferences": ["preference1", "preference2", "preference3", "preference4"],
    "motivation": "...",
    "challenges": ["challenge1", "challenge2", "challenge3"]
  }},
  "context_analysis": {{
    "environment": "...",
    "duration": "...",
    "constraints": ["constraint1", "constraint2", "constraint3"],
    "resources": ["resource1", "resource2", "resource3"],
    "technical_requirements": ["requirement1", "requirement2"]
  }},
  "task_analysis": {{
    "main_topics": ["topic1", "topic2", "topic3"],
    "subtopics": ["subtopic1", "subtopic2", "subtopic3", "subtopic4", "subtopic5", "subtopic6"],
    "prerequisites": ["prerequisite1", "prerequisite2"],
    "review_summary": "..."
  }}
}}
```

Output JSON only."""


@tool
def run_analysis(
    title: str,
    target_audience: str,
    learning_environment: str,
    duration: str,
    prior_knowledge: Optional[str],
    learning_goals: list[str],
) -> dict:
    """
    Perform the ADDIE Analysis phase. (sub-items 1-10)

    Returns:
        The analysis section of the standard schema
    """
    llm = get_llm()

    prompt = ANALYSIS_PROMPT.format(
        title=title,
        target_audience=target_audience,
        learning_environment=learning_environment,
        duration=duration,
        prior_knowledge=prior_knowledge or "Not specified",
        learning_goals=json.dumps(learning_goals, ensure_ascii=False),
    )

    try:
        response = llm.invoke(prompt)
        return parse_json_response(response.content)
    except Exception as e:
        print(f"[WARN] run_analysis failed: {e}")
        return _fallback_analysis(target_audience, learning_environment, duration, learning_goals)


def _fallback_analysis(target_audience, learning_environment, duration, learning_goals):
    """Analysis 폴백"""
    return {
        "needs_analysis": {
            "problem_definition": f"Systematic training is needed to improve the competencies of {target_audience}",
            "gap_analysis": [f"Current: lack of {g} competency → Target: achieve {g}" for g in learning_goals[:3]],
            "performance_analysis": "Judged to be a competency gap that can be closed through training",
            "priority_matrix": {"high": learning_goals[:2], "medium": learning_goals[2:3], "low": []},
        },
        "learner_analysis": {
            "target_audience": target_audience,
            "characteristics": ["Motivated to learn", "Has basic knowledge", "Interested in practical application", "Capable of self-directed learning", "Prefers collaboration"],
            "prior_knowledge": "Has basic-level related knowledge",
            "learning_preferences": ["Hands-on oriented", "Step-by-step guidance", "Visual materials", "Immediate feedback"],
            "motivation": "High motivation to improve work/academic performance",
            "challenges": ["Time constraints", "Lack of practice opportunities", "Individual differences"],
        },
        "context_analysis": {
            "environment": learning_environment,
            "duration": duration,
            "constraints": ["Time constraints", "Limited resources", "Technical environment"],
            "resources": ["Learning platform", "Course materials", "Mentoring"],
            "technical_requirements": ["Internet connection", "Learning device"],
        },
        "task_analysis": {
            "main_topics": learning_goals[:3],
            "subtopics": [f"{g} fundamentals" for g in learning_goals[:3]] + [f"{g} practice" for g in learning_goals[:3]],
            "prerequisites": ["Understanding of basic concepts", "How to use the learning tools"],
            "review_summary": f"Analysis complete. A {duration} course for {target_audience}, planned to achieve {len(learning_goals)} objectives.",
        },
    }


# ============================================================
# 2. Design 단계 (소항목 11-18)
# ============================================================

DESIGN_PROMPT = """You are an instructional design expert. Based on the following analysis results, perform the ADDIE Design phase.

## Scenario Information
- Title: {title}
- Target learners: {target_audience}
- Duration: {duration}
- Learning goals: {learning_goals}

## Design Phase Requirements (8 sub-items)

### D-11. Learning objective elaboration (learning_objectives)
- At least 5 measurable learning objectives
- Distributed across Bloom's Taxonomy levels (2 at Remember/Understand, 2 at Apply/Analyze, 1 at Evaluate/Create)
- Each objective must include: id, level, statement, bloom_verb, measurable

### D-12. Assessment planning (assessment_plan)
- diagnostic: At least 2 diagnostic assessment methods
- formative: At least 2 formative assessment methods
- summative: At least 2 summative assessment methods

### D-13. Instructional content selection (content_structure)
- modules: At least 3 modules
- topics: Topics for each module
- sequencing: Description of the learning sequence

### D-14. Instructional strategy design (instructional_strategies)
- model: "Gagné's 9 Events"
- sequence: Include all 9 Events (required!)
- methods: At least 3 instructional methods
- rationale: Rationale for the selected strategy

### D-15. Non-instructional strategies (non_instructional_strategies)
- motivation_strategies: 2-3 motivational strategies
- self_directed_learning: 2-3 self-directed learning supports
- support_strategies: Other support measures

### D-16. Media selection (media_selection)
- Select at least 3 media appropriate for each learning activity

### D-17. Learning activities and time structuring (learning_activities)
- Design at least 3 learning activities

### D-18. Storyboard (storyboard)
- Design at least 3 frames

## Gagné's 9 Events (all must be included)
1. Gain attention, 2. Inform learners of the objectives, 3. Stimulate recall of prior learning, 4. Present the content,
5. Provide learning guidance, 6. Elicit performance, 7. Provide feedback, 8. Assess performance, 9. Enhance retention and transfer

## Output Format (JSON)
```json
{{
  "learning_objectives": [
    {{"id": "LO-01", "level": "Remember", "statement": "...", "bloom_verb": "define", "measurable": true}},
    {{"id": "LO-02", "level": "Understand", "statement": "...", "bloom_verb": "explain", "measurable": true}},
    {{"id": "LO-03", "level": "Apply", "statement": "...", "bloom_verb": "apply", "measurable": true}},
    {{"id": "LO-04", "level": "Analyze", "statement": "...", "bloom_verb": "analyze", "measurable": true}},
    {{"id": "LO-05", "level": "Evaluate", "statement": "...", "bloom_verb": "evaluate", "measurable": true}}
  ],
  "assessment_plan": {{
    "diagnostic": ["diagnostic1", "diagnostic2"],
    "formative": ["formative1", "formative2"],
    "summative": ["summative1", "summative2"]
  }},
  "content_structure": {{
    "modules": ["module1", "module2", "module3"],
    "topics": ["topic1", "topic2", "topic3"],
    "sequencing": "..."
  }},
  "instructional_strategies": {{
    "model": "Gagné's 9 Events",
    "sequence": [
      {{"event": "Gain attention", "activity": "...", "duration": "5 min", "resources": ["..."]}},
      {{"event": "Inform learners of the objectives", "activity": "...", "duration": "5 min", "resources": ["..."]}},
      {{"event": "Stimulate recall of prior learning", "activity": "...", "duration": "10 min", "resources": ["..."]}},
      {{"event": "Present the content", "activity": "...", "duration": "20 min", "resources": ["..."]}},
      {{"event": "Provide learning guidance", "activity": "...", "duration": "10 min", "resources": ["..."]}},
      {{"event": "Elicit performance", "activity": "...", "duration": "15 min", "resources": ["..."]}},
      {{"event": "Provide feedback", "activity": "...", "duration": "10 min", "resources": ["..."]}},
      {{"event": "Assess performance", "activity": "...", "duration": "15 min", "resources": ["..."]}},
      {{"event": "Enhance retention and transfer", "activity": "...", "duration": "10 min", "resources": ["..."]}}
    ],
    "methods": ["method1", "method2", "method3"],
    "rationale": "..."
  }},
  "non_instructional_strategies": {{
    "motivation_strategies": ["motivation1", "motivation2"],
    "self_directed_learning": ["self-directed1", "self-directed2"],
    "support_strategies": ["support1", "support2"]
  }},
  "learning_activities": [
    {{"activity_name": "activity1", "duration": "20 min", "description": "...", "materials": ["..."]}},
    {{"activity_name": "activity2", "duration": "30 min", "description": "...", "materials": ["..."]}},
    {{"activity_name": "activity3", "duration": "20 min", "description": "...", "materials": ["..."]}}
  ],
  "media_selection": [
    {{"media_type": "media1", "purpose": "...", "rationale": "..."}},
    {{"media_type": "media2", "purpose": "...", "rationale": "..."}},
    {{"media_type": "media3", "purpose": "...", "rationale": "..."}}
  ],
  "storyboard": [
    {{"frame_number": 1, "screen_title": "...", "visual_description": "...", "interaction": "...", "notes": "..."}},
    {{"frame_number": 2, "screen_title": "...", "visual_description": "...", "interaction": "...", "notes": "..."}},
    {{"frame_number": 3, "screen_title": "...", "visual_description": "...", "interaction": "...", "notes": "..."}}
  ]
}}
```

Output JSON only."""


@tool
def run_design(
    title: str,
    target_audience: str,
    duration: str,
    learning_goals: list[str],
) -> dict:
    """
    Perform the ADDIE Design phase. (sub-items 11-18)

    Returns:
        The design section of the standard schema
    """
    llm = get_llm()

    prompt = DESIGN_PROMPT.format(
        title=title,
        target_audience=target_audience,
        duration=duration,
        learning_goals=json.dumps(learning_goals, ensure_ascii=False),
    )

    try:
        response = llm.invoke(prompt)
        return parse_json_response(response.content)
    except Exception as e:
        print(f"[WARN] run_design failed: {e}")
        return _fallback_design(learning_goals, duration)


def _fallback_design(learning_goals, duration):
    """Design 폴백"""
    objectives = [
        {"id": f"LO-0{i+1}", "level": ["Remember", "Understand", "Apply", "Analyze", "Evaluate"][i % 5],
         "statement": f"Can perform {g}", "bloom_verb": ["define", "explain", "apply", "analyze", "evaluate"][i % 5],
         "measurable": True}
        for i, g in enumerate(learning_goals[:5])
    ]

    events = ["Gain attention", "Inform learners of the objectives", "Stimulate recall of prior learning", "Present the content", "Provide learning guidance",
              "Elicit performance", "Provide feedback", "Assess performance", "Enhance retention and transfer"]

    return {
        "learning_objectives": objectives,
        "assessment_plan": {
            "diagnostic": ["Pre-quiz", "Self-checklist"],
            "formative": ["Practice assignment", "Peer assessment"],
            "summative": ["Final assessment", "Project presentation"],
        },
        "content_structure": {
            "modules": learning_goals[:3],
            "topics": [f"{g} details" for g in learning_goals[:3]],
            "sequencing": "Fundamentals → advanced → application order",
        },
        "instructional_strategies": {
            "model": "Gagné's 9 Events",
            "sequence": [{"event": e, "activity": f"{e} activity", "duration": "10 min", "resources": ["Course materials"]} for e in events],
            "methods": ["Lecture", "Hands-on practice", "Discussion"],
            "rationale": "Gagné's 9 Events applied for systematic instructional design",
        },
        "non_instructional_strategies": {
            "motivation_strategies": ["Recognition of achievement", "Connection to practice"],
            "self_directed_learning": ["Learning checklist", "Self-assessment"],
            "support_strategies": ["Mentoring", "Q&A"],
        },
        "learning_activities": [
            {"activity_name": "Concept learning", "duration": "30 min", "description": "Learn the core concepts", "materials": ["Course materials"]},
            {"activity_name": "Hands-on practice", "duration": "40 min", "description": "Complete the practice assignment", "materials": ["Practice materials"]},
            {"activity_name": "Discussion", "duration": "20 min", "description": "Group discussion", "materials": ["Discussion topics"]},
        ],
        "media_selection": [
            {"media_type": "Slides", "purpose": "Convey concepts", "rationale": "Supports visual learning"},
            {"media_type": "Video", "purpose": "Demonstration", "rationale": "Conveys procedural knowledge"},
            {"media_type": "Practice environment", "purpose": "Application", "rationale": "Experiential learning"},
        ],
        "storyboard": [
            {"frame_number": i+1, "screen_title": f"Screen {i+1}", "visual_description": "Learning content", "interaction": "Next button", "notes": ""}
            for i in range(3)
        ],
    }


# ============================================================
# 3. Development 단계 (소항목 17, 19-23)
# ============================================================

DEVELOPMENT_PROMPT = """You are an instructional design expert. Perform the ADDIE Development phase.

## Scenario Information
- Title: {title}
- Target learners: {target_audience}
- Learning environment: {learning_environment}
- Learning goals: {learning_goals}

## Development Phase Requirements (5 sub-items)

### Dev-19. Learner material development (learner_materials)
- At least 3 learner materials
- Each material must include: title, type, content (500+ characters of actual content), format

### Dev-20. Instructor manual (instructor_guide)
- overview: Overall overview
- session_guides: At least 3 per-session guides
- facilitation_tips: At least 3 facilitation tips
- troubleshooting: At least 2 troubleshooting guides

### Dev-21. Operator manual (operator_manual)
- system_setup: System setup guide
- operation_procedures: Operating procedures
- support_procedures: Support procedures
- escalation_process: Escalation process

### Dev-22. Assessment tools/items (assessment_tools)
- At least 10 items (3-4 easy, 4-5 medium, 2-3 hard)
- Each item: item_id, type, question, options (4), answer, aligned_objective, scoring_criteria

### Dev-23. Expert review (expert_review)
- reviewers: At least 2 reviewer types
- review_criteria: At least 5 review criteria
- feedback_summary: Feedback summary
- revisions_made: Revisions made

## Output Format (JSON)
```json
{{
  "learner_materials": [
    {{"title": "material1", "type": "Handout", "content": "Detailed content...", "format": "PDF"}},
    {{"title": "material2", "type": "Slides", "content": "Detailed content...", "format": "PPT"}},
    {{"title": "material3", "type": "Practice guide", "content": "Detailed content...", "format": "PDF"}}
  ],
  "instructor_guide": {{
    "overview": "Overview of the whole course...",
    "session_guides": [
      {{"session": 1, "objectives": ["objective1"], "activities": ["activity1"], "notes": "..."}},
      {{"session": 2, "objectives": ["objective2"], "activities": ["activity2"], "notes": "..."}},
      {{"session": 3, "objectives": ["objective3"], "activities": ["activity3"], "notes": "..."}}
    ],
    "facilitation_tips": ["tip1", "tip2", "tip3"],
    "troubleshooting": ["troubleshooting1", "troubleshooting2"]
  }},
  "operator_manual": {{
    "system_setup": "System setup guide...",
    "operation_procedures": ["procedure1", "procedure2", "procedure3"],
    "support_procedures": ["support1", "support2"],
    "escalation_process": "Escalation procedure..."
  }},
  "assessment_tools": [
    {{"item_id": "Q-01", "type": "Multiple choice", "question": "item1", "options": ["A", "B", "C", "D"], "answer": "A", "aligned_objective": "LO-01", "scoring_criteria": "1 point for a correct answer"}},
    {{"item_id": "Q-02", "type": "Multiple choice", "question": "item2", "options": ["A", "B", "C", "D"], "answer": "B", "aligned_objective": "LO-02", "scoring_criteria": "1 point for a correct answer"}}
  ],
  "expert_review": {{
    "reviewers": ["Subject matter expert", "Instructional design expert"],
    "review_criteria": ["Content accuracy", "Instructional appropriateness", "Technical quality", "User experience", "Accessibility"],
    "feedback_summary": "Generally good, but some improvements are needed...",
    "revisions_made": ["Content revision1", "Design improvement2"]
  }}
}}
```

Output JSON only."""


@tool
def run_development(
    title: str,
    target_audience: str,
    learning_environment: str,
    learning_goals: list[str],
) -> dict:
    """
    ADDIE Development 단계를 수행합니다. (소항목 17, 19-23)

    Returns:
        표준 스키마의 development 섹션
    """
    llm = get_llm()

    prompt = DEVELOPMENT_PROMPT.format(
        title=title,
        target_audience=target_audience,
        learning_environment=learning_environment,
        learning_goals=json.dumps(learning_goals, ensure_ascii=False),
    )

    try:
        response = llm.invoke(prompt)
        return parse_json_response(response.content)
    except Exception as e:
        print(f"[WARN] run_development failed: {e}")
        return _fallback_development(learning_goals)


def _fallback_development(learning_goals):
    """Development 폴백"""
    return {
        "learner_materials": [
            {"title": f"Learning material {i+1}", "type": ["handout", "slides", "practice guide"][i % 3],
             "content": f"Detailed learning content for the learning objective '{learning_goals[i % len(learning_goals)]}'. " * 10,
             "format": "PDF"}
            for i in range(3)
        ],
        "instructor_guide": {
            "overview": "This course is designed to improve learners' competencies.",
            "session_guides": [
                {"session": i+1, "objectives": [learning_goals[i % len(learning_goals)]], "activities": ["lecture", "practice"], "notes": "Points to note"}
                for i in range(3)
            ],
            "facilitation_tips": ["Encourage learner participation", "Time management", "Encourage questions"],
            "troubleshooting": ["Use alternatives in case of technical problems", "Give additional explanation when learners struggle"],
        },
        "operator_manual": {
            "system_setup": "LMS access and content registration procedure",
            "operation_procedures": ["Pre-check", "Operational monitoring", "Post-session wrap-up"],
            "support_procedures": ["Respond to learner enquiries", "Technical support"],
            "escalation_process": "Escalate in the order coordinator -> manager -> technical team",
        },
        "assessment_tools": [
            {"item_id": f"Q-{i+1:02d}", "type": "multiple choice", "question": f"Item {i+1}",
             "options": ["A", "B", "C", "D"], "answer": ["A", "B", "C", "D"][i % 4],
             "aligned_objective": f"LO-{(i % 5) + 1:02d}", "scoring_criteria": "1 point for a correct answer"}
            for i in range(10)
        ],
        "expert_review": {
            "reviewers": ["subject matter expert", "instructional design expert"],
            "review_criteria": ["Content accuracy", "Instructional appropriateness", "Technical quality", "User experience", "Accessibility"],
            "feedback_summary": "Satisfactory overall",
            "revisions_made": ["Revised some content", "Improved the design"],
        },
    }


# ============================================================
# 4. Implementation 단계 (소항목 24-27)
# ============================================================

IMPLEMENTATION_PROMPT = """You are an instructional design expert. Perform the ADDIE Implementation phase.

## Scenario Information
- Title: {title}
- Target learners: {target_audience}
- Learning environment: {learning_environment}
- Class size: {class_size}

## Implementation Phase Requirements (4 sub-items)

### I-24. Instructor/operator orientation (instructor_orientation)
- orientation_objectives: at least 3 orientation objectives
- schedule: schedule plan
- materials: at least 2 required materials
- competency_checklist: at least 3 competency checklist items

### I-25. System/environment check (system_check)
- checklist: at least 5 check items
- technical_validation: technical validation results
- contingency_plans: at least 2 contingency plans

### I-26. Prototype execution plan (prototype_execution)
- pilot_scope: pilot scope
- participants: participant size and characteristics
- execution_log: at least 3 execution log entries
- issues_encountered: at least 2 issues encountered

### I-27. Operational monitoring (monitoring)
- monitoring_criteria: at least 3 monitoring criteria
- support_channels: at least 2 support channels
- issue_resolution_log: at least 2 issue-resolution records
- real_time_adjustments: at least 2 real-time adjustments

## Output Format (JSON)
```json
{{
  "instructor_orientation": {{
    "orientation_objectives": ["objective1", "objective2", "objective3"],
    "schedule": "Schedule plan description...",
    "materials": ["material1", "material2"],
    "competency_checklist": ["competency1", "competency2", "competency3"]
  }},
  "system_check": {{
    "checklist": ["check1", "check2", "check3", "check4", "check5"],
    "technical_validation": "Technical validation complete...",
    "contingency_plans": ["contingency1", "contingency2"]
  }},
  "prototype_execution": {{
    "pilot_scope": "Pilot scope description...",
    "participants": "Participant information...",
    "execution_log": ["log1", "log2", "log3"],
    "issues_encountered": ["issue1", "issue2"]
  }},
  "monitoring": {{
    "monitoring_criteria": ["criterion1", "criterion2", "criterion3"],
    "support_channels": ["channel1", "channel2"],
    "issue_resolution_log": ["resolution1", "resolution2"],
    "real_time_adjustments": ["adjustment1", "adjustment2"]
  }}
}}
```

Output JSON only."""


@tool
def run_implementation(
    title: str,
    target_audience: str,
    learning_environment: str,
    class_size: Optional[int],
) -> dict:
    """
    ADDIE Implementation 단계를 수행합니다. (소항목 24-27)

    Returns:
        표준 스키마의 implementation 섹션
    """
    llm = get_llm()

    prompt = IMPLEMENTATION_PROMPT.format(
        title=title,
        target_audience=target_audience,
        learning_environment=learning_environment,
        class_size=class_size or "not specified",
    )

    try:
        response = llm.invoke(prompt)
        return parse_json_response(response.content)
    except Exception as e:
        print(f"[WARN] run_implementation failed: {e}")
        return _fallback_implementation(learning_environment)


def _fallback_implementation(learning_environment):
    """Implementation 폴백"""
    return {
        "instructor_orientation": {
            "orientation_objectives": ["Understand the course", "Use the tools", "Manage learners"],
            "schedule": "A 2-hour orientation one week before the course starts",
            "materials": ["Instructor guide", "System manual"],
            "competency_checklist": ["Content understanding", "Tool usage", "Facilitation skills"],
        },
        "system_check": {
            "checklist": ["LMS access", "Content loading", "Assessment function", "Communication tools", "Backup system"],
            "technical_validation": "All functions confirmed to work correctly",
            "contingency_plans": ["Prepare offline materials", "Technical support contact"],
        },
        "prototype_execution": {
            "pilot_scope": f"Run a small-scale pilot in the {learning_environment} environment",
            "participants": "10-15 people selected from the target learners",
            "execution_log": ["Pilot started", "Mid-point check", "Pilot finished"],
            "issues_encountered": ["Some technical problems", "Schedule adjustment needed"],
        },
        "monitoring": {
            "monitoring_criteria": ["Learning progress", "Participation", "Satisfaction"],
            "support_channels": ["Real-time chat", "Email support"],
            "issue_resolution_log": ["Resolved technical problems", "Provided learning support"],
            "real_time_adjustments": ["Schedule adjustment", "Provided additional explanation"],
        },
    }


# ============================================================
# 5. Evaluation 단계 (소항목 22, 28-33)
# ============================================================

EVALUATION_PROMPT = """You are an instructional design expert. Perform the ADDIE Evaluation phase.

## Scenario Information
- Title: {title}
- Target learners: {target_audience}
- Learning goals: {learning_goals}

## Evaluation Phase Requirements (6 sub-items)

### E-28. Pilot data collection (formative.data_collection)
- methods: at least 3 collection methods
- learner_feedback: at least 3 learner-feedback items
- performance_data: performance data metrics
- observations: at least 2 observation items

### E-29. Formative-evaluation-based improvements (formative.improvements)
- at least 3 improvement items
- each item: issue_identified, improvement_action, priority

### E-30. Summative assessment items (summative.assessment_tools)
- at least 5 summative assessment items
- each item: item_id, type, question, scoring_rubric

### E-31. Summative effectiveness analysis (summative.effectiveness_analysis)
- learning_outcomes: learning-outcome analysis
- goal_achievement_rate: goal achievement rate
- statistical_analysis: statistical analysis results
- recommendations: at least 3 recommendations

### E-32. Program adoption decision (summative.adoption_decision)
- decision: choose one of adopt/modify/reject
- rationale: grounds for the decision (2-3 sentences)
- conditions: at least 2 adoption conditions
- stakeholder_approval: stakeholder approval status

### E-33. Program improvement and feedback loop (improvement_plan)
- feedback_summary: feedback summary
- improvement_areas: at least 3 improvement areas
- action_items: at least 3 action items
- feedback_loop: description of the feedback loop
- next_iteration_goals: at least 2 goals for the next iteration

## Output Format (JSON)
```json
{{
  "formative": {{
    "data_collection": {{
      "methods": ["method1", "method2", "method3"],
      "learner_feedback": ["feedback1", "feedback2", "feedback3"],
      "performance_data": {{"metric1": "indicator1", "metric2": "indicator2"}},
      "observations": ["observation1", "observation2"]
    }},
    "improvements": [
      {{"issue_identified": "issue1", "improvement_action": "improvement1", "priority": "high"}},
      {{"issue_identified": "issue2", "improvement_action": "improvement2", "priority": "medium"}},
      {{"issue_identified": "issue3", "improvement_action": "improvement3", "priority": "low"}}
    ]
  }},
  "summative": {{
    "assessment_tools": [
      {{"item_id": "SA-01", "type": "comprehensive assessment", "question": "item1", "scoring_rubric": "scoring criteria1"}},
      {{"item_id": "SA-02", "type": "comprehensive assessment", "question": "item2", "scoring_rubric": "scoring criteria2"}}
    ],
    "effectiveness_analysis": {{
      "learning_outcomes": {{"achievement_rate": "85%", "details": "Outcome analysis..."}},
      "goal_achievement_rate": "85%",
      "statistical_analysis": "Statistical analysis results...",
      "recommendations": ["recommendation1", "recommendation2", "recommendation3"]
    }},
    "adoption_decision": {{
      "decision": "adopt",
      "rationale": "Explanation of the grounds for adoption...",
      "conditions": ["condition1", "condition2"],
      "stakeholder_approval": "Approved"
    }}
  }},
  "improvement_plan": {{
    "feedback_summary": "Feedback summary...",
    "improvement_areas": ["area1", "area2", "area3"],
    "action_items": ["action1", "action2", "action3"],
    "feedback_loop": "Description of the feedback loop...",
    "next_iteration_goals": ["goal1", "goal2"]
  }}
}}
```

Output JSON only."""


@tool
def run_evaluation(
    title: str,
    target_audience: str,
    learning_goals: list[str],
) -> dict:
    """
    ADDIE Evaluation 단계를 수행합니다. (소항목 22, 28-33)

    Returns:
        표준 스키마의 evaluation 섹션
    """
    llm = get_llm()

    prompt = EVALUATION_PROMPT.format(
        title=title,
        target_audience=target_audience,
        learning_goals=json.dumps(learning_goals, ensure_ascii=False),
    )

    try:
        response = llm.invoke(prompt)
        return parse_json_response(response.content)
    except Exception as e:
        print(f"[WARN] run_evaluation failed: {e}")
        return _fallback_evaluation(learning_goals)


def _fallback_evaluation(learning_goals):
    """Evaluation 폴백"""
    return {
        "formative": {
            "data_collection": {
                "methods": ["Survey", "Learning-log analysis", "Interview"],
                "learner_feedback": ["Content satisfaction", "Appropriateness of difficulty", "Practicality"],
                "performance_data": {"completion_rate": "Course completion rate", "assessment_score": "Assessment score"},
                "observations": ["Learning participation", "Interaction frequency"],
            },
            "improvements": [
                {"issue_identified": "Some content is difficult", "improvement_action": "Provide additional explanation", "priority": "high"},
                {"issue_identified": "Insufficient time", "improvement_action": "Adjust the schedule", "priority": "medium"},
                {"issue_identified": "Insufficient practice", "improvement_action": "Extend practice time", "priority": "high"},
            ],
        },
        "summative": {
            "assessment_tools": [
                {"item_id": f"SA-{i+1:02d}", "type": "comprehensive assessment", "question": f"Comprehensive assessment item {i+1}",
                 "scoring_rubric": "Scoring criteria"}
                for i in range(5)
            ],
            "effectiveness_analysis": {
                "learning_outcomes": {"achievement_rate": "85%", "details": "Most learning objectives achieved"},
                "goal_achievement_rate": "85%",
                "statistical_analysis": "Mean score 85, standard deviation 10",
                "recommendations": ["Supplement the content", "Strengthen practice", "Improve feedback"],
            },
            "adoption_decision": {
                "decision": "adopt",
                "rationale": "The goal achievement rate is high and learner satisfaction is good",
                "conditions": ["Apply after revising some content", "Reflect the pilot results"],
                "stakeholder_approval": "Approved",
            },
        },
        "improvement_plan": {
            "feedback_summary": "Positive overall, but some improvement is needed",
            "improvement_areas": ["Content difficulty", "Practice opportunities", "Feedback system"],
            "action_items": ["Revise content", "Add practice", "Strengthen feedback"],
            "feedback_loop": "Quarterly review and improvement cycle",
            "next_iteration_goals": ["Reach 90% satisfaction", "Reach 95% completion rate"],
        },
    }


__all__ = [
    "run_analysis",
    "run_design",
    "run_development",
    "run_implementation",
    "run_evaluation",
]
