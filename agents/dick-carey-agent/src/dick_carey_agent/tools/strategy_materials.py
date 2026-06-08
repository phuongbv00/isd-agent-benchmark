"""
Strategy & Materials Tools (Steps 6-7)

Dick & Carey Model's Steps 6-7:
6. Instructional Strategy Development
7. Instructional Materials Development
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


# ========== Step 6: Instructional Strategy Development ==========
@tool
def develop_instructional_strategy(
    performance_objectives: dict,
    learner_analysis: dict,
    learning_environment: str,
    duration: str,
) -> dict:
    """
    Develop instructional strategy. (Dick & Carey Step 6)

    Design instructional strategies for achieving learning objectives.
    Consists of pre-instructional activities, content presentation, learner participation, and assessment.

    Args:
        performance_objectives: Performance objectives
        learner_analysis: Learner analysis results
        learning_environment: Learning environment
        duration: Learning duration

    Returns:
        Instructional strategy result (pre_instructional, content_presentation, learner_participation, assessment, delivery_method, grouping_strategy)
    """
    try:
        llm = get_llm()

        prompt = f"""You are an expert in the Dick & Carey model. Develop an effective Instructional Strategy.

## Input Information
- Performance Objectives: {json.dumps(performance_objectives, ensure_ascii=False)}
- Learner Analysis: {json.dumps(learner_analysis, ensure_ascii=False)}
- Learning Environment: {learning_environment}
- Duration: {duration}

## Dick & Carey's Instructional Strategy Components
1. Pre-instructional Activities
   - Motivation
   - Objectives
   - Prerequisite Skills

2. Content Presentation & Learning Guidance
   - Sequence
   - Examples & Non-examples

3. Learner Participation
   - Practice
   - Feedback

4. Assessment & Follow-through
   - Assessment Strategy
   - Retention & Transfer

## Required Elements
1. pre_instructional: Pre-instructional activities (motivation, objectives_info, prerequisite_review)
2. content_presentation: Content presentation (sequence, examples, non_examples, practice_guidance)
3. learner_participation: Learner participation (practice_activities minimum 3, feedback_strategy)
4. assessment: Assessment strategy (assessment_strategy, retention_transfer)
5. delivery_method: Delivery method
6. grouping_strategy: Grouping strategy
7. content_selection (D-13): Content selection (core_content, supplementary_content, selection_rationale)
8. non_instructional_strategy (D-15): Non-instructional strategy development (strategies, rationale, implementation)
9. media_selection (D-16): Media selection and utilization plan (selected_media, selection_criteria, utilization_plan)

## Output Format (JSON)
```json
{{
  "pre_instructional": {{
    "motivation": "Present application cases from actual work situations to help recognize the necessity and usefulness of learning. Motivate through successful case examples.",
    "objectives_info": "Clearly present performance objectives before learning begins, and guide specific outcomes to achieve after learning.",
    "prerequisite_review": "Confirm prior learning through brief pre-quiz. Provide supplementary materials for deficient areas."
  }},
  "content_presentation": {{
    "sequence": [
      "1. Introduction and definition of core concepts",
      "2. Explanation of principles and theoretical background",
      "3. Presentation of practical application cases",
      "4. Step-by-step procedure guidance",
      "5. Practice and application"
    ],
    "examples": [
      "Successful application cases in actual work situations",
      "Demonstrations showing step-by-step procedures",
      "Application examples in various situations"
    ],
    "non_examples": [
      "Incorrect application cases and their results",
      "Common mistake patterns"
    ],
    "practice_guidance": "Provide immediate practice opportunities after concept explanation. Gradually increase difficulty through practice."
  }},
  "learner_participation": {{
    "practice_activities": [
      "Concept confirmation quiz: Check understanding of core concepts",
      "Case analysis activity: Analyze and discuss presented cases",
      "Practice assignments: Application practice in real situations",
      "Collaborative project: Team problem-solving activities"
    ],
    "feedback_strategy": "Provide immediate and specific feedback. Guide improvement direction beyond just right/wrong. Utilize peer feedback."
  }},
  "assessment": {{
    "assessment_strategy": "Combine formative and summative assessment. Regularly check understanding during learning, conduct comprehensive evaluation after learning.",
    "retention_transfer": "Assign tasks to apply learning content to actual work. Provide regular review opportunities. Run application case sharing sessions."
  }},
  "delivery_method": "{learning_environment}",
  "grouping_strategy": "Appropriately mix individual learning, small group collaborative learning (3-5 people), and whole class discussion",
  "content_selection": {{
    "core_content": ["Core concepts and principles", "Basic procedures and methods", "Key application cases"],
    "supplementary_content": ["Advanced theory", "Advanced techniques", "Reference materials"],
    "selection_rationale": "Prioritize core content essential for achieving learning objectives, and organize supplementary content considering learner level and time constraints."
  }},
  "non_instructional_strategy": {{
    "strategies": ["Provide work manuals and guides", "Establish mentoring/coaching system", "Improve performance management and feedback system", "Improve work environment"],
    "rationale": "Non-instructional support strategies to complement environmental and organizational factors that cannot be solved by training alone",
    "implementation": "Build support systems that can be immediately applied in the workplace alongside training"
  }},
  "media_selection": {{
    "selected_media": ["Presentation", "Video", "Practice environment", "Print materials"],
    "selection_criteria": "Selected considering learning objective type, learning environment, learner characteristics, and cost efficiency",
    "utilization_plan": "Motivate with video in introduction, combine presentation and practice in main content, distribute handouts for wrap-up"
  }}
}}
```

Output JSON only."""

        response = llm.invoke(prompt)
        return parse_json_response(response.content)
    except Exception:
        return _fallback_develop_instructional_strategy(performance_objectives, learner_analysis, learning_environment, duration)


def _fallback_develop_instructional_strategy(
    performance_objectives: dict,
    learner_analysis: dict,
    learning_environment: str,
    duration: str,
) -> dict:
    """Fallback function when LLM fails"""
    return {
        "pre_instructional": {
            "motivation": "Present practical application cases to help recognize the necessity of learning. Motivate through success stories.",
            "objectives_info": "Clearly present learning objectives and guide achievement criteria.",
            "prerequisite_review": "Confirm prior learning through pre-quiz. Provide supplementary materials if deficient.",
        },
        "content_presentation": {
            "sequence": [
                "1. Core concept introduction",
                "2. Theoretical background explanation",
                "3. Application case presentation",
                "4. Step-by-step procedure guidance",
                "5. Practice and application",
            ],
            "examples": ["Successful application cases", "Step-by-step demonstrations", "Various application examples"],
            "non_examples": ["Incorrect application cases", "Common mistake patterns"],
            "practice_guidance": "Practice immediately after concept explanation. Gradually increase difficulty.",
        },
        "learner_participation": {
            "practice_activities": [
                "Concept confirmation quiz",
                "Case analysis and discussion",
                "Practice assignment completion",
                "Collaborative project",
            ],
            "feedback_strategy": "Provide immediate and specific feedback. Guide improvement direction. Utilize peer feedback.",
        },
        "assessment": {
            "assessment_strategy": "Combine formative and summative assessment. Check understanding during learning.",
            "retention_transfer": "Actual work application tasks. Regular review. Application case sharing.",
        },
        "delivery_method": learning_environment,
        "grouping_strategy": "Mix individual learning, small groups (3-5 people), and whole class discussion",
        # D-13: Content selection
        "content_selection": {
            "core_content": ["Core concepts and principles", "Basic procedures and methods", "Key application cases"],
            "supplementary_content": ["Advanced theory", "Advanced techniques", "Reference materials"],
            "selection_rationale": "Prioritize core content essential for achieving learning objectives, and organize supplementary content considering learner level and time constraints.",
        },
        # D-15: Non-instructional strategy development
        "non_instructional_strategy": {
            "strategies": [
                "Provide work manuals and guides",
                "Establish mentoring/coaching system",
                "Improve performance management and feedback system",
                "Improve work environment",
            ],
            "rationale": "Non-instructional support strategies to complement environmental and organizational factors that cannot be solved by training alone",
            "implementation": "Build support systems that can be immediately applied in the workplace alongside training",
        },
        # D-16: Media selection and utilization plan
        "media_selection": {
            "selected_media": ["Presentation", "Video", "Practice environment", "Print materials"],
            "selection_criteria": "Selected considering learning objective type, learning environment, learner characteristics, and cost efficiency",
            "utilization_plan": "Motivate with video in introduction, combine presentation and practice in main content, distribute handouts for wrap-up",
        },
    }


# ========== Step 7: Instructional Materials Development ==========
@tool
def develop_instructional_materials(
    instructional_strategy: dict,
    performance_objectives: dict,
    learning_environment: str,
    duration: str,
    topic_title: str,
) -> dict:
    """
    Develop instructional materials. (Dick & Carey Step 7)

    Develop instructor guide, learner materials, and media materials according to instructional strategy.

    Args:
        instructional_strategy: Instructional strategy
        performance_objectives: Performance objectives
        learning_environment: Learning environment
        duration: Learning duration
        topic_title: Topic title

    Returns:
        Instructional materials result (instructor_guide, learner_materials, media_list, slide_contents)
    """
    try:
        llm = get_llm()

        prompt = f"""You are an expert in the Dick & Carey model. Develop effective Instructional Materials.

## Input Information
- Topic: {topic_title}
- Instructional Strategy: {json.dumps(instructional_strategy, ensure_ascii=False)}
- Performance Objectives: {json.dumps(performance_objectives, ensure_ascii=False)}
- Learning Environment: {learning_environment}
- Duration: {duration}

## Dick & Carey's Instructional Materials Development Principles
1. Instructional materials are tools that implement instructional strategy
2. Learner-centered design
3. Materials that support goal achievement
4. Format suitable for learning environment

## Required Elements
1. instructor_guide: Instructor guide (detailed facilitation guidance, including storyboard)
2. learner_materials: Learner materials **minimum 3 types** (storyboard can be included for each)
3. media_list: Media materials list (each with storyboard)
4. slide_contents: PPT slides **minimum 10**
5. instructor_manual (Dev-20): Instructor manual development (detailed lesson facilitation guide, problem situation response, etc.)
6. operator_manual (Dev-21): Operator manual development (facility preparation, participant management, operation checklist, etc.)
7. expert_review (Dev-23): Expert review (reviewer, review_date, review_areas, findings, recommendations, approval_status)

## Output Format (JSON)
- slide_contents: Minimum 10 slides (introduction-main content-wrap-up structure)
- instructor_manual: Markdown format, include sections (preparation/lesson facilitation/assessment feedback/problem response)
- operator_manual: Markdown format, include sections (training preparation/day-of operation/post-processing/emergency response)

```json
{{
  "instructor_guide": {{
    "type": "Instructor Guide",
    "title": "[Topic] Lesson Facilitation Guide",
    "description": "Detailed facilitation guide for instructors",
    "content_outline": ["Lesson overview", "Introduction facilitation", "Main content facilitation", "Activity facilitation", "Wrap-up"],
    "pages": 15,
    "duration": "[Duration]"
  }},
  "learner_materials": [
    {{"type": "Learner Workbook", "title": "[Topic] Learning Workbook", "description": "Learning content summary and practice problems", "content_outline": ["Concept summary", "Practice problems", "Practice assignments"], "pages": 20}},
    {{"type": "Handout", "title": "[Topic] Key Summary", "description": "Key concept summary", "content_outline": ["Core concepts", "Key procedures"], "pages": 2}},
    {{"type": "Practice Guide", "title": "[Topic] Practice Guide", "description": "Step-by-step practice guide", "content_outline": ["Practice objectives", "Step-by-step procedures"], "pages": 8}}
  ],
  "media_list": [
    {{"type": "Presentation", "title": "[Topic] PPT", "description": "Lesson PPT", "duration": "[Duration]"}},
    {{"type": "Video", "title": "[Topic] Concept Video", "description": "Core concept explanation video", "duration": "10 min"}}
  ],
  "slide_contents": [
    {{"slide_number": 1, "title": "Learning Guide", "bullet_points": ["Learning topic", "Learning objectives", "Learning sequence"], "speaker_notes": "Guide overall flow", "visual_suggestion": "Display title and objectives"}},
    {{"slide_number": 5, "title": "Core Concepts", "bullet_points": ["Concept definition", "Key features", "Examples"], "speaker_notes": "Explain core concepts", "visual_suggestion": "Concept diagram"}},
    {{"slide_number": 10, "title": "Next Steps", "bullet_points": ["Assignment guidance", "Next learning preview"], "speaker_notes": "Wrap-up and assignment guidance", "visual_suggestion": "Checklist"}}
  ],
  "instructor_manual": "# [Topic] Instructor Manual\\n\\n## 1. Preparation\\n...\\n\\n## 2. Lesson Facilitation Guide\\n...\\n\\n## 3. Assessment and Feedback\\n...\\n\\n## 4. Problem Situation Response\\n...",
  "operator_manual": "# [Topic] Operator Manual\\n\\n## 1. Training Preparation\\n...\\n\\n## 2. Day-of Operation\\n...\\n\\n## 3. Post-processing\\n...\\n\\n## 4. Emergency Response\\n...",
  "expert_review": {{
    "reviewer": "Instructional design expert",
    "review_date": "Within 1 week after development completion",
    "review_areas": ["Content accuracy", "Objective-content alignment", "Strategy appropriateness"],
    "findings": ["Key review findings"],
    "recommendations": ["Improvement recommendations"],
    "approval_status": "Approval status"
  }}
}}
```

**Required Elements**:
1. slide_contents must generate all **10 slides** from slide_number 1 to 10 (above example is for format reference)
2. instructor_manual and operator_manual should have detailed content in each section

Output JSON only."""

        response = llm.invoke(prompt)
        return parse_json_response(response.content)
    except Exception:
        return _fallback_develop_instructional_materials(instructional_strategy, performance_objectives, learning_environment, duration, topic_title)


def _fallback_develop_instructional_materials(
    instructional_strategy: dict,
    performance_objectives: dict,
    learning_environment: str,
    duration: str,
    topic_title: str,
) -> dict:
    """Fallback function when LLM fails"""
    # D-18: Storyboard/screen flow design
    storyboard = [
        {
            "frame_number": 1,
            "screen_title": "Introduction Screen",
            "visual_description": "Display training title and logo, welcome message",
            "audio_narration": "Welcome to the training.",
            "interaction": "Click start button",
            "notes": "Background music fade in",
        },
        {
            "frame_number": 2,
            "screen_title": "Learning Objectives",
            "visual_description": "Learning objectives list animation",
            "audio_narration": "Let's review today's learning content and objectives.",
            "interaction": "Auto-advance",
            "notes": "Display objectives sequentially",
        },
        {
            "frame_number": 3,
            "screen_title": "Core Content 1",
            "visual_description": "Core concept diagram",
            "audio_narration": "We will learn the first core content.",
            "interaction": "Click next button",
            "notes": "Provide step-by-step explanation",
        },
    ]

    return {
        "instructor_guide": {
            "type": "Instructor Guide",
            "title": f"{topic_title} Lesson Facilitation Guide",
            "description": "Detailed facilitation guide for instructors",
            "content_outline": [
                "Lesson overview",
                "Introduction facilitation",
                "Main content facilitation",
                "Activity facilitation",
                "Wrap-up",
            ],
            "pages": 15,
            "duration": duration,
            "storyboard": storyboard,
        },
        "learner_materials": [
            {
                "type": "Learner Workbook",
                "title": f"{topic_title} Learning Workbook",
                "description": "Includes learning content summary and practice problems",
                "content_outline": ["Concept summary", "Practice problems", "Practice assignments"],
                "pages": 20,
                "duration": "",
                "storyboard": [],
            },
            {
                "type": "Handout",
                "title": f"{topic_title} Key Summary",
                "description": "Key concept summary material",
                "content_outline": ["Core concepts", "Key procedures"],
                "pages": 2,
                "duration": "",
                "storyboard": [],
            },
            {
                "type": "Practice Guide",
                "title": f"{topic_title} Practice Guide",
                "description": "Step-by-step practice guide",
                "content_outline": ["Practice objectives", "Step-by-step procedures", "Evaluation criteria"],
                "pages": 8,
                "duration": "",
                "storyboard": [],
            },
        ],
        "media_list": [
            {
                "type": "Presentation",
                "title": f"{topic_title} PPT",
                "description": "Lesson PPT",
                "content_outline": [],
                "pages": 0,
                "duration": duration,
                "storyboard": storyboard,
            },
            {
                "type": "Video",
                "title": f"{topic_title} Concept Video",
                "description": "Core concept explanation video",
                "content_outline": [],
                "pages": 0,
                "duration": "10 min",
                "storyboard": [
                    {
                        "frame_number": 1,
                        "screen_title": "Opening",
                        "visual_description": "Logo and title",
                        "audio_narration": "We will explain the core concepts.",
                        "interaction": "None",
                        "notes": "Background music",
                    },
                ],
            },
        ],
        "slide_contents": [
            {"slide_number": i, "title": f"Slide {i}", "bullet_points": ["Content 1", "Content 2", "Content 3"], "speaker_notes": f"Slide {i} explanation", "visual_suggestion": "Related image"}
            for i in range(1, 11)
        ],
        # Dev-20: Instructor manual development
        "instructor_manual": f"""# {topic_title} Instructor Manual

## 1. Preparation
- Review and familiarize with training materials
- Equipment and environment check
- Learner information confirmation

## 2. Lesson Facilitation Guide
### Introduction (10 min)
- Welcome learners and ice-breaking
- Guide learning objectives and schedule
- Confirm prior learning

### Main Content (40 min)
- Core concept explanation (using presentation)
- Case presentation and discussion facilitation
- Practice activity facilitation (individual/group)

### Wrap-up (10 min)
- Key content summary
- Q&A and feedback
- Follow-up learning guidance

## 3. Assessment and Feedback
- Formative assessment implementation method
- Feedback provision guidelines
- Performance recording and reporting

## 4. Problem Situation Response
- Alternatives for technical issues
- Response to low learner participation
- Adjustment plans when time is short""",
        # Dev-21: Operator manual development
        "operator_manual": f"""# {topic_title} Operator Manual

## 1. Training Preparation
### Facility Preparation
- Classroom reservation and setup
- Equipment check (projector, audio, PC)
- Learning materials printing and arrangement

### Participant Management
- Participant list confirmation
- Attendance check system preparation
- Name tag and materials distribution preparation

## 2. Day-of Operation
### Operation Checklist
- [ ] Classroom temperature and lighting check
- [ ] Final equipment test
- [ ] Refreshments preparation
- [ ] Emergency contact network confirmation

### Facilitation Support
- Attendance check and name tag distribution
- Time management and break announcements
- Unexpected situation response

## 3. Post-processing
- Classroom cleanup and equipment return
- Attendance record compilation and report
- Satisfaction survey collection
- Cost settlement

## 4. Emergency Response
- Backup equipment location for equipment failure
- Contacts for emergency situations
- Procedures for training cancellation/postponement""",
        # Dev-23: Expert review
        "expert_review": {
            "reviewer": "Instructional design expert (SME)",
            "review_date": "Within 1 week after development completion",
            "review_areas": [
                "Content accuracy and currency",
                "Alignment between learning objectives and content",
                "Appropriateness of instructional strategy",
                "Validity of assessment tools",
                "Quality and completeness of materials",
            ],
            "findings": [
                "Overall good alignment between learning objectives and content",
                "Practice activities effective for achieving learning objectives",
                "Some terminology and expressions need revision",
            ],
            "recommendations": [
                "Update introduction cases with more recent examples",
                "Recommend adding 10 minutes to practice time",
                "Review assessment item difficulty adjustment",
            ],
            "approval_status": "Conditional approval - applicable after minor revisions",
        },
    }
