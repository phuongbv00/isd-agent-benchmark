"""
Analysis Stage Tools

ADDIE's first stage: Learner, context, and task analysis
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
def analyze_learner(
    target_audience: str,
    prior_knowledge: Optional[str] = None,
    additional_context: Optional[str] = None,
) -> dict:
    """
    Perform learner analysis.

    Args:
        target_audience: Target learners (e.g., "new employees", "5th grade students")
        prior_knowledge: Prior knowledge level (optional)
        additional_context: Additional context information (optional)

    Returns:
        Learner analysis results (5+ characteristics, 2-3 sentence motivation, 3+ challenges)
    """
    try:
        llm = get_llm()

        prompt = f"""You are an instructional design expert with 20 years of experience. Perform an in-depth analysis of the following target learners.

## Input Information
- Target Audience: {target_audience}
- Prior Knowledge Level: {prior_knowledge or "Not specified"}
- Additional Context: {additional_context or "Not specified"}

## Analysis Items (Minimum Requirements)
1. characteristics: Learner characteristics **minimum 5** (cognitive, affective, social characteristics)
2. learning_preferences: Learning preferences **minimum 4**
3. motivation: Motivation level **2-3 sentences** with detailed explanation
4. challenges: Anticipated difficulties **minimum 3**

## Output Format (JSON)
```json
{{
  "target_audience": "{target_audience}",
  "characteristics": [
    "Currently adapting to new environment and needs to learn organizational culture",
    "Shows high interest in practical application and strong desire for rapid growth",
    "Prefers learning through actual work cases rather than theory",
    "Has strong desire for relationship building with colleagues",
    "Generation familiar with digital tool utilization"
  ],
  "prior_knowledge": "Has basic theoretical knowledge as a related major graduate but lacks practical experience. Recognizes the gap between academic learning and actual work.",
  "learning_preferences": [
    "Hands-on experiential learning",
    "Step-by-step structured guidance",
    "Collaborative learning with peers",
    "Immediate feedback provision"
  ],
  "motivation": "Has strong intrinsic motivation for organizational adaptation and competency development, while also having extrinsic motivation to demonstrate quick results. Has desire to clearly understand their role and contribute to the team.",
  "challenges": [
    "Potential confusion due to information overload",
    "Frustration from recognizing theory-practice gap",
    "Difficulty understanding organizational expectations"
  ]
}}
```

Output JSON only."""

        response = llm.invoke(prompt)
        return parse_json_response(response.content)
    except Exception:
        return _fallback_analyze_learner(target_audience, prior_knowledge, additional_context)


def _fallback_analyze_learner(
    target_audience: str,
    prior_knowledge: Optional[str] = None,
    additional_context: Optional[str] = None,
) -> dict:
    """Fallback function when LLM fails"""
    audience_lower = target_audience.lower()

    characteristics = []
    learning_preferences = []
    motivation = "Has moderate learning motivation."
    challenges = []

    if "new" in audience_lower or "beginner" in audience_lower or "entry" in audience_lower:
        characteristics = [
            "Currently adapting to new environment and needs organizational culture learning",
            "High interest in practical application and strong desire for rapid growth",
            "Prefers learning through actual work cases rather than theory",
            "Strong desire for relationship building with colleagues",
            "Familiar with digital tool utilization",
        ]
        learning_preferences = ["Hands-on practice", "Step-by-step guidance", "Collaborative learning", "Immediate feedback"]
        motivation = "Has strong intrinsic motivation for organizational adaptation and competency development. Also has extrinsic motivation to demonstrate quick results."
        challenges = ["Information overload", "Theory-practice gap", "Understanding expectations"]

    elif "elementary" in audience_lower or "primary" in audience_lower or "child" in audience_lower:
        characteristics = [
            "Highly curious and exploratory",
            "Short attention span (15-20 minutes)",
            "Prefers play-based learning",
            "Heavily influenced by peer groups",
            "Transitioning from concrete operational to formal operational stage",
        ]
        learning_preferences = ["Visual materials", "Game-based", "Group activities", "Reward systems"]
        motivation = "Tends to rely on extrinsic motivation, but intrinsic motivation also emerges for interesting topics."
        challenges = ["Understanding abstract concepts", "Long-term concentration", "Difficulty with self-directed learning"]

    elif "professional" in audience_lower or "adult" in audience_lower or "employee" in audience_lower:
        characteristics = [
            "Busy schedule with time constraints",
            "Values practical application",
            "Capable of self-directed learning",
            "Connects rich experience to learning",
            "Prefers clear learning objectives",
        ]
        learning_preferences = ["Efficient learning", "Case-based", "Immediate application", "Flexible schedule"]
        motivation = "Strong motivation for work performance improvement and career development. Values practical value of learning."
        challenges = ["Securing learning time", "Changing existing habits", "Generational technology gap"]

    else:
        characteristics = [
            "Possesses general learning ability",
            "Diverse background knowledge",
            "Individual differences exist",
            "Basic learning motivation",
            "Adaptable",
        ]
        learning_preferences = ["Various methods", "Visual materials", "Combined practice", "Feedback"]
        challenges = ["Need to consider individual differences", "Maintaining motivation", "Adjusting content difficulty"]

    return {
        "target_audience": target_audience,
        "characteristics": characteristics,
        "prior_knowledge": prior_knowledge or "Has basic prior knowledge",
        "learning_preferences": learning_preferences,
        "motivation": motivation,
        "challenges": challenges,
    }


@tool
def analyze_context(
    learning_environment: str,
    duration: str,
    class_size: Optional[int] = None,
    budget: Optional[str] = None,
    resources: Optional[list[str]] = None,
) -> dict:
    """
    Analyze learning environment.

    Args:
        learning_environment: Learning environment (e.g., "online", "in-person classroom", "blended")
        duration: Learning duration (e.g., "2 hours", "1 day", "4 weeks")
        class_size: Number of learners (optional)
        budget: Budget level (optional)
        resources: List of available resources (optional)

    Returns:
        Context analysis results (3+ constraints, 3+ resources, 2+ technical requirements)
    """
    try:
        llm = get_llm()

        prompt = f"""You are an instructional design expert with 20 years of experience. Perform an in-depth analysis of the following learning environment.

## Input Information
- Learning Environment: {learning_environment}
- Learning Duration: {duration}
- Number of Learners: {class_size or "Not specified"}
- Budget: {budget or "Not specified"}
- Available Resources: {json.dumps(resources, ensure_ascii=False) if resources else "Not specified"}

## Analysis Items (Minimum Requirements)
1. constraints: Constraints **minimum 3**
2. resources: Available resources **minimum 3**
3. technical_requirements: Technical requirements **minimum 2**

## Output Format (JSON)
```json
{{
  "environment": "{learning_environment}",
  "duration": "{duration}",
  "constraints": [
    "Internet connection stability required",
    "Requires learner self-management ability",
    "Non-verbal communication limitations"
  ],
  "resources": [
    "Screen sharing capability",
    "Real-time chat",
    "Recording feature",
    "Breakout rooms for small groups"
  ],
  "technical_requirements": [
    "Video conferencing platform (Zoom/Teams/Meet)",
    "Stable internet connection (minimum 10Mbps)"
  ]
}}
```

Output JSON only."""

        response = llm.invoke(prompt)
        return parse_json_response(response.content)
    except Exception:
        return _fallback_analyze_context(learning_environment, duration, class_size, budget, resources)


def _fallback_analyze_context(
    learning_environment: str,
    duration: str,
    class_size: Optional[int] = None,
    budget: Optional[str] = None,
    resources: Optional[list[str]] = None,
) -> dict:
    """Fallback function when LLM fails"""
    env_lower = learning_environment.lower()

    constraints = []
    technical_requirements = []
    available_resources = resources or []

    if "online" in env_lower:
        constraints = ["Internet connection required", "Self-management ability required", "Potential technical issues"]
        technical_requirements = ["Video conferencing tool", "LMS access", "Stable internet"]
        if not available_resources:
            available_resources = ["Screen sharing", "Chat", "Recording", "Breakout rooms"]

    elif "in-person" in env_lower or "classroom" in env_lower or "face-to-face" in env_lower:
        constraints = ["Physical space required", "Travel time", "Schedule coordination"]
        technical_requirements = ["Projector/Screen", "Whiteboard", "Audio system"]
        if not available_resources:
            available_resources = ["Textbooks", "Presentations", "Practice materials", "Writing tools"]

    elif "blended" in env_lower or "hybrid" in env_lower:
        constraints = ["Coordination between both environments needed", "Technology gap", "Maintaining consistency"]
        technical_requirements = ["Video conferencing", "LMS", "Classroom equipment", "Internet"]
        if not available_resources:
            available_resources = ["Online content", "In-person materials", "Recorded videos"]

    else:
        constraints = ["Environment characteristics need identification", "Resource constraints", "Time constraints"]
        technical_requirements = ["Basic equipment", "Learning materials"]
        if not available_resources:
            available_resources = ["Textbooks", "Presentations"]

    if class_size and class_size > 30:
        constraints.append("Large group management required")

    return {
        "environment": learning_environment,
        "duration": duration,
        "constraints": constraints,
        "resources": available_resources,
        "technical_requirements": technical_requirements,
    }


@tool
def analyze_task(
    learning_goals: list[str],
    domain: Optional[str] = None,
    difficulty: Optional[str] = None,
) -> dict:
    """
    Analyze learning tasks.

    Args:
        learning_goals: List of learning goals
        domain: Educational domain (e.g., "IT", "Business", "Language")
        difficulty: Difficulty level (e.g., "easy", "medium", "hard")

    Returns:
        Task analysis results (3+ main topics, 6+ subtopics, 2+ prerequisites)
    """
    try:
        llm = get_llm()

        prompt = f"""You are an instructional design expert with 20 years of experience. Perform task analysis for the following learning goals.

## Input Information
- Learning Goals: {json.dumps(learning_goals, ensure_ascii=False)}
- Educational Domain: {domain or "General"}
- Difficulty: {difficulty or "medium"}

## Analysis Items (Minimum Requirements)
1. main_topics: Main topics **minimum 3**
2. subtopics: Subtopics **minimum 6** (at least 2 per main topic)
3. prerequisites: Prerequisites **minimum 2**

## Output Format (JSON)
```json
{{
  "main_topics": [
    "Understanding organizational culture and values",
    "Work processes and systems",
    "Team collaboration and communication"
  ],
  "subtopics": [
    "Understanding company vision and mission",
    "Organizational structure and department roles",
    "Core work procedure familiarization",
    "Internal system usage",
    "Effective work communication",
    "Team collaboration methods and tools"
  ],
  "prerequisites": [
    "Basic computer skills",
    "Understanding of basic business terminology"
  ],
  "knowledge_structure": {{
    "foundation": "Basic concepts and terminology",
    "core": "Core content and principles",
    "application": "Practice and application"
  }},
  "review_summary": "Based on this task analysis, learners need to study 3 main topics and 6 subtopics, proceeding in the order of foundation → core → application. Systematic learning is possible for learners who meet the prerequisite requirements."
}}
```

Output JSON only."""

        response = llm.invoke(prompt)
        return parse_json_response(response.content)
    except Exception:
        return _fallback_analyze_task(learning_goals, domain, difficulty)


def _fallback_analyze_task(
    learning_goals: list[str],
    domain: Optional[str] = None,
    difficulty: Optional[str] = None,
) -> dict:
    """Fallback function when LLM fails"""
    main_topics = learning_goals[:3] if len(learning_goals) >= 3 else learning_goals + ["Additional learning topic"] * (3 - len(learning_goals))

    subtopics = []
    for topic in main_topics:
        subtopics.append(f"{topic} - Basic concepts")
        subtopics.append(f"{topic} - Practice and application")

    prerequisites = ["Basic knowledge", "Learning motivation"]

    if domain:
        domain_lower = domain.lower()
        if "it" in domain_lower or "programming" in domain_lower or "tech" in domain_lower:
            prerequisites = ["Computer basics", "Logical thinking skills"]
        elif "business" in domain_lower or "management" in domain_lower:
            prerequisites = ["Basic business terminology", "Organizational understanding"]
        elif "language" in domain_lower or "english" in domain_lower:
            prerequisites = ["Basic vocabulary", "Basic grammar understanding"]

    return {
        "main_topics": main_topics,
        "subtopics": subtopics[:6],
        "prerequisites": prerequisites,
        "knowledge_structure": {
            "foundation": "Basic concepts and terminology",
            "core": "Core content and principles",
            "application": "Practice and application",
        },
        "review_summary": f"Based on this task analysis, learners need to study {len(main_topics)} main topics and {len(subtopics[:6])} subtopics, proceeding in the order of foundation → core → application. Systematic learning is possible for learners who meet the prerequisite requirements ({len(prerequisites)}).",
    }


@tool
def analyze_needs(
    learning_goals: list[str],
    current_state: Optional[str] = None,
    desired_state: Optional[str] = None,
    performance_gap: Optional[str] = None,
) -> dict:
    """
    Perform needs analysis (Gap Analysis).

    Args:
        learning_goals: List of learning goals
        current_state: Current state description (optional)
        desired_state: Desired state description (optional)
        performance_gap: Performance gap description (optional)

    Returns:
        Needs analysis results (gap_analysis, root_causes, training_needs, non_training_solutions)
    """
    try:
        llm = get_llm()

        prompt = f"""You are an instructional design expert with 20 years of experience. Perform needs analysis based on the following information.

## Input Information
- Learning Goals: {json.dumps(learning_goals, ensure_ascii=False)}
- Current State: {current_state or "Not specified"}
- Desired State: {desired_state or "Not specified"}
- Performance Gap: {performance_gap or "Not specified"}

## Needs Analysis Items (Required)
1. **gap_analysis**: Analysis of differences between current and target state (3+ items)
2. **root_causes**: Root causes of performance issues (3+ items)
3. **training_needs**: Needs addressable through training (3+ items)
4. **non_training_solutions**: Solutions beyond training (2+ items)
5. **priority**: Priority level (high/medium/low)
6. **recommendation**: Judgment on appropriateness of training solution and recommendations

## Output Format (JSON)
```json
{{
  "gap_analysis": [
    "Current: Lacking basic job knowledge → Target: Capable of independent work performance",
    "Current: Insufficient understanding of organizational culture → Target: Internalization of organizational values and culture",
    "Current: Unfamiliar with collaboration tools → Target: Proficient use of collaboration tools"
  ],
  "root_causes": [
    "Lack of systematic onboarding training",
    "Insufficient practical learning opportunities",
    "Inadequate mentoring system"
  ],
  "training_needs": [
    "Organizational culture and vision training",
    "Work process hands-on training",
    "Collaboration tool training"
  ],
  "non_training_solutions": [
    "Mentoring program operation",
    "Work manual and guide provision"
  ],
  "priority": "high",
  "priority_matrix": {{
    "high_urgency_high_impact": ["Organizational culture and vision training", "Work process hands-on"],
    "high_urgency_low_impact": ["Basic collaboration tool training"],
    "low_urgency_high_impact": ["Advanced competency development"],
    "low_urgency_low_impact": ["Optional self-development courses"]
  }},
  "recommendation": "Systematic onboarding training is essential for rapid adaptation and productivity improvement of new employees. It is recommended to combine training with mentoring program to support practical adaptation."
}}
```

Output JSON only."""

        response = llm.invoke(prompt)
        return parse_json_response(response.content)
    except Exception:
        return _fallback_analyze_needs(learning_goals, current_state, desired_state, performance_gap)


def _fallback_analyze_needs(
    learning_goals: list[str],
    current_state: Optional[str] = None,
    desired_state: Optional[str] = None,
    performance_gap: Optional[str] = None,
) -> dict:
    """Fallback function when LLM fails"""
    # Generate gap analysis based on learning goals
    gap_analysis = []
    for goal in learning_goals[:3]:
        gap_analysis.append(f"Current: Lacking competency in {goal} → Target: Achievement of {goal}")

    if len(gap_analysis) < 3:
        gap_analysis.extend([
            "Current: Lacking basic knowledge → Target: Understanding core concepts",
            "Current: Difficulty in practical application → Target: Applicable to real situations",
            "Current: Insufficient self-directed learning → Target: Development of independent learning ability",
        ][:3 - len(gap_analysis)])

    root_causes = [
        "Lack of systematic training program",
        "Insufficient practice and application opportunities",
        "Inadequate feedback and coaching system",
    ]

    training_needs = [f"Training related to {goal}" for goal in learning_goals[:3]]
    if len(training_needs) < 3:
        training_needs.extend([
            "Basic competency strengthening training",
            "Practical application workshop",
            "Self-directed learning methods training",
        ][:3 - len(training_needs)])

    non_training_solutions = [
        "Job guides and manuals provision",
        "Mentoring/coaching program operation",
    ]

    # Generate priority matrix
    priority_matrix = {
        "high_urgency_high_impact": training_needs[:2] if len(training_needs) >= 2 else training_needs,
        "high_urgency_low_impact": [training_needs[2]] if len(training_needs) > 2 else [],
        "low_urgency_high_impact": ["Advanced competency development"],
        "low_urgency_low_impact": ["Optional self-development courses"],
    }

    return {
        "gap_analysis": gap_analysis,
        "root_causes": root_causes,
        "training_needs": training_needs,
        "non_training_solutions": non_training_solutions,
        "priority": "medium",
        "priority_matrix": priority_matrix,
        "recommendation": "A systematic training program is needed to achieve learning goals. It is recommended to provide practical application opportunities and continuous feedback along with training.",
    }
