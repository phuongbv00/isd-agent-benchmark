"""
Implementation Stage Tools

ADDIE's fourth stage: Implementation plan development
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
def create_implementation_plan(
    lesson_plan: dict,
    learning_environment: str,
    target_audience: str,
    class_size: Optional[int] = None,
) -> dict:
    """
    Develop implementation plan.

    Args:
        lesson_plan: Lesson plan
        learning_environment: Learning environment
        target_audience: Target learners
        class_size: Number of learners (optional)

    Returns:
        Implementation plan (facilitator_guide 200+ chars, learner_guide 200+ chars, technical_requirements 2+, support_plan)
    """
    try:
        llm = get_llm()

        prompt = f"""You are an instructional design expert with 20 years of experience. Develop a detailed implementation plan based on the following information.

## Input Information
- Lesson Plan: {json.dumps(lesson_plan, ensure_ascii=False)}
- Learning Environment: {learning_environment}
- Target Audience: {target_audience}
- Number of Learners: {class_size or "Not specified"}

## Requirements
1. **facilitator_guide**: **200+ characters** detailed guide (step-by-step numbers, time allocation, facilitation instructions)
2. **learner_guide**: **200+ characters** detailed guide (before/during/after learning guidance)
3. **technical_requirements**: **Minimum 2** technical requirements
4. **support_plan**: Learner support plan

## Output Format (JSON)
```json
{{
  "delivery_method": "In-person classroom training",
  "facilitator_guide": "1. Pre-preparation (10 min before training): Check classroom, test projector and audio, arrange learning materials, prepare attendance sheet\\n2. Opening (5 min): Welcome participants, present today's learning objectives, introduce schedule, icebreaking activity\\n3. Module 1 facilitation (30 min): Explain slides then group discussion, circulate tables to facilitate discussion, summarize key opinions on whiteboard\\n4. Break (10 min): Announce break time, provide refreshments\\n5. Module 2-3 facilitation: Focus on hands-on practice, provide individual support to struggling learners\\n6. Wrap-up (10 min): Summarize key content, Q&A, announce satisfaction survey",
  "learner_guide": "1. Pre-learning preparation: Complete pre-survey, set personal learning goals, prepare note-taking tools, attend in comfortable attire\\n2. During learning participation: Active questioning and sharing opinions, collaborate with peers in group activities, try hands-on practice yourself, ask questions immediately when unclear\\n3. Post-learning activities: Review handouts, develop workplace application plan, share learning content with colleagues, complete satisfaction survey",
  "operator_guide": "1. Training environment preparation: Confirm classroom reservation, check equipment (projector, microphone, PC), print and arrange learning materials, prepare refreshments\\n2. Participant management: Take attendance, distribute name tags, guide seating, record special notes\\n3. Operations support: Communicate with instructor, manage time, announce break times, respond to unexpected situations\\n4. Post-processing: Clean up classroom, return equipment, collect surveys, report attendance status",
  "orientation_plan": "1. Instructor/Facilitator orientation (1 week before training): Explain training objectives and curriculum, deliver and review teaching materials, discuss facilitation approach, Q&A\\n2. Operator orientation (3 days before training): Explain operations roles and responsibilities, distribute checklist, share emergency contacts, confirm rehearsal schedule\\n3. Rehearsal (1 day before training): Equipment testing, confirm flow, check time allocation, final coordination",
  "pilot_plan": {{
    "pilot_scope": "1st Pilot: Full course trial with small group (10-15 people)",
    "participants": "1-2 representatives from each department, training staff observers",
    "duration": "Same as main training (2 hours)",
    "success_criteria": ["Learning objective achievement rate 80% or higher", "Satisfaction 4.0/5.0 or higher", "No major issues during execution"],
    "data_collection": ["Pre/post test scores", "Satisfaction survey", "Observation records", "Participant feedback"],
    "contingency_plan": "Use backup materials for technical issues, reduce optional modules if time exceeds"
  }},
  "technical_requirements": [
    "Projector and screen (1080p resolution or higher)",
    "Audio system (microphone, speakers)",
    "Whiteboard and markers",
    "PC or tablet for each participant (for practice)"
  ],
  "support_plan": "Assign 1 assistant facilitator during training for practice support, operate Q&A channel for 1 week after training, send additional learning materials via email, maintain practice environment access for 1 month"
}}
```

Output JSON only."""

        response = llm.invoke(prompt)
        return parse_json_response(response.content)
    except Exception:
        return _fallback_create_implementation_plan(lesson_plan, learning_environment, target_audience, class_size)


def _fallback_create_implementation_plan(
    lesson_plan: dict,
    learning_environment: str,
    target_audience: str,
    class_size: Optional[int] = None,
) -> dict:
    """Fallback function when LLM fails"""
    env_lower = learning_environment.lower()
    is_online = "online" in env_lower

    if is_online:
        delivery_method = "Online live training"
        facilitator_guide = """1. Pre-preparation (15 min before training): Connect to video conferencing platform, test screen sharing, check chat and breakout room functions, start recording
2. Opening (5 min): Welcome participants and verify audio/video, present today's learning objectives, explain participation methods (chat, raise hand function)
3. Main training facilitation: Proceed with slides via screen sharing, ask engagement questions every 10 minutes, use breakout rooms for discussion, monitor chat
4. Wrap-up (10 min): Summarize key content, Q&A, announce recording and material sharing, send survey link"""

        learner_guide = """1. Pre-learning preparation: Install and test video conferencing platform connection, secure quiet learning space, prepare webcam and microphone, review pre-materials
2. During learning participation: Keep camera on and actively participate, ask questions via chat or raise hand, actively participate in breakout activities, notify via chat if having difficulty concentrating
3. Post-learning activities: Review recording, download provided materials, complete practice assignments, email additional questions"""

        operator_guide = """1. Platform preparation: Create and distribute video conference link, confirm recording settings, pre-configure breakout rooms, set up waiting room
2. Participant management: Monitor connection status, support connection issues, manage chat, take and record attendance
3. Technical support: Resolve screen sharing issues, support audio issues, prepare backup link, respond to network failures
4. Post-processing: Edit and upload recording, organize attendance records, collect surveys, send participant emails"""

        orientation_plan = """1. Instructor/Facilitator orientation (3 days before training): Platform feature familiarization, screen sharing test, breakout room operation guidance, teaching material review
2. Operator orientation (2 days before training): Explain technical support role, share troubleshooting manual, confirm emergency contacts, distribute checklist
3. Rehearsal (1 day before training): Full flow test, check backup scenarios, confirm time allocation, complete final check"""

        technical_requirements = [
            "Video conferencing platform (Zoom/Teams/Meet)",
            "Stable internet connection (minimum 10Mbps)",
            "Webcam and microphone",
            "LMS access rights",
        ]

        support_plan = "Assign chat monitoring staff during training, provide phone support for technical issues, share recording within 24 hours, operate Q&A board for 1 week"

    else:
        delivery_method = "In-person classroom training"
        facilitator_guide = """1. Pre-preparation (10 min before training): Check classroom, test projector and audio, arrange learning materials, prepare attendance sheet, set up refreshments
2. Opening (5 min): Welcome participants, present today's learning objectives, introduce schedule, icebreaking activity to set atmosphere
3. Main training facilitation: Combine slide explanation with group discussion, circulate tables to facilitate discussion, provide individual support during practice, respond immediately to questions
4. Wrap-up (10 min): Summarize key content, Q&A, distribute satisfaction survey, announce follow-up learning"""

        learner_guide = """1. Pre-learning preparation: Complete pre-survey, set personal learning goals, prepare note-taking tools, attend in comfortable attire, prepare business cards (for networking)
2. During learning participation: Active questioning and sharing opinions, collaborate with peers in group activities, learn by doing hands-on practice yourself, ask questions immediately when unclear
3. Post-learning activities: Review handouts, develop workplace application plan, share learning content with colleagues, complete satisfaction survey"""

        operator_guide = """1. Training environment preparation: Confirm classroom reservation, check equipment (projector, microphone, PC), print and arrange learning materials, prepare refreshments
2. Participant management: Take attendance, distribute name tags, guide seating, record special notes
3. Operations support: Communicate with instructor, manage time, announce break times, respond to unexpected situations
4. Post-processing: Clean up classroom, return equipment, collect surveys, report attendance status"""

        orientation_plan = """1. Instructor/Facilitator orientation (1 week before training): Explain training objectives and curriculum, deliver and review teaching materials, discuss facilitation approach, Q&A
2. Operator orientation (3 days before training): Explain operations roles and responsibilities, distribute checklist, share emergency contacts, confirm rehearsal schedule
3. Rehearsal (1 day before training): Equipment testing, confirm flow, check time allocation, final coordination"""

        technical_requirements = [
            "Projector and screen",
            "Audio system (microphone, speakers)",
            "Whiteboard and markers",
            "Participant practice PCs (if needed)",
        ]

        support_plan = "Assign assistant facilitator during training for practice support, handle individual questions during breaks, send additional materials via email after training, provide Q&A email support for 1 week"

    # Pilot plan
    pilot_plan = {
        "pilot_scope": "1st Pilot: Full course trial with small group (10-15 people)",
        "participants": "1-2 representatives from each department, training staff observers",
        "duration": "Same as main training",
        "success_criteria": ["Learning objective achievement rate 80% or higher", "Satisfaction 4.0/5.0 or higher", "No major issues during execution"],
        "data_collection": ["Pre/post test scores", "Satisfaction survey", "Observation records", "Participant feedback"],
        "contingency_plan": "Use backup materials for technical issues, reduce optional modules if time exceeds",
    }

    return {
        "delivery_method": delivery_method,
        "facilitator_guide": facilitator_guide,
        "learner_guide": learner_guide,
        "operator_guide": operator_guide,
        "orientation_plan": orientation_plan,
        "pilot_plan": pilot_plan,
        "technical_requirements": technical_requirements,
        "support_plan": support_plan,
    }


@tool
def create_maintenance_plan(
    program_title: str,
    delivery_method: str,
    content_types: list[str],
    update_frequency: Optional[str] = None,
) -> dict:
    """
    Develop training program maintenance plan.

    Args:
        program_title: Training program title
        delivery_method: Training delivery method (in-person/online/blended)
        content_types: Content type list (e.g., slides, videos, handouts)
        update_frequency: Update frequency (optional, default: quarterly)

    Returns:
        Maintenance plan (content_maintenance, technical_maintenance, quality_assurance, version_control)
    """
    try:
        llm = get_llm()

        prompt = f"""You are an instructional design expert with 20 years of experience. Develop a training program maintenance plan.

## Input Information
- Program Title: {program_title}
- Delivery Method: {delivery_method}
- Content Types: {json.dumps(content_types, ensure_ascii=False)}
- Update Frequency: {update_frequency or "Quarterly"}

## Requirements
1. **content_maintenance**: Content maintenance plan (update procedures, responsible parties, frequency)
2. **technical_maintenance**: Technical maintenance (systems, platforms, tools)
3. **quality_assurance**: Quality management plan (review process, feedback incorporation)
4. **version_control**: Version management and history tracking

## Output Format (JSON)
```json
{{
  "program_title": "{program_title}",
  "maintenance_period": "1 year (auto-renewal)",
  "content_maintenance": {{
    "review_cycle": "Quarterly review",
    "update_triggers": [
      "Immediate update upon regulation/policy change",
      "Incorporate learner feedback (monthly)",
      "Industry trend changes (quarterly)",
      "Improvement based on evaluation results (semi-annually)"
    ],
    "update_process": [
      "1. Receive and review change request",
      "2. Impact analysis and priority determination",
      "3. Content modification and internal review",
      "4. SME verification",
      "5. Final approval and deployment",
      "6. Record change history"
    ],
    "responsible": {{
      "content_owner": "Instructional Design Team",
      "sme": "Subject Matter Experts",
      "reviewer": "Quality Assurance Team",
      "approver": "Training Operations Manager"
    }}
  }},
  "technical_maintenance": {{
    "platform_updates": "Monthly regular inspection and updates",
    "backup_policy": "Daily automatic backup, weekly full backup",
    "security_review": "Quarterly security inspection",
    "performance_monitoring": "Real-time monitoring and monthly reports",
    "disaster_recovery": "Establish disaster recovery plan and annual DR drill"
  }},
  "quality_assurance": {{
    "review_checklist": [
      "Learning objective alignment verification",
      "Content accuracy verification",
      "Accessibility standards compliance",
      "User experience (UX) inspection",
      "Technical error testing"
    ],
    "feedback_channels": [
      "Post-learning satisfaction survey",
      "Real-time feedback button",
      "Regular FGI (Focus Group Interview)",
      "Line manager feedback"
    ],
    "improvement_cycle": "Collect feedback → Analysis → Derive improvements → Apply → Measure effectiveness"
  }},
  "version_control": {{
    "naming_convention": "v[Major].[Minor].[Patch]_YYYYMMDD",
    "major_version": "Curriculum structure change, learning objective change",
    "minor_version": "Content addition/modification, feature improvement",
    "patch_version": "Typo correction, bug fix",
    "changelog_management": "Record all changes and share with stakeholders",
    "archive_policy": "Retain previous versions for 2 years"
  }},
  "support_resources": {{
    "helpdesk": "Weekdays 09:00-18:00 operation",
    "self_service": "FAQ, user guide, video tutorials",
    "escalation_path": "1st Helpdesk → 2nd Technical Support Team → 3rd Development Team"
  }}
}}
```

Output JSON only."""

        response = llm.invoke(prompt)
        return parse_json_response(response.content)
    except Exception:
        return _fallback_create_maintenance_plan(program_title, delivery_method, content_types, update_frequency)


def _fallback_create_maintenance_plan(
    program_title: str,
    delivery_method: str,
    content_types: list[str],
    update_frequency: Optional[str] = None,
) -> dict:
    """Fallback function when LLM fails"""
    return {
        "program_title": program_title,
        "maintenance_period": "1 year (auto-renewal)",
        "content_maintenance": {
            "review_cycle": update_frequency or "Quarterly review",
            "update_triggers": [
                "Immediate update upon regulation/policy change",
                "Incorporate learner feedback (monthly)",
                "Industry trend changes (quarterly)",
                "Improvement based on evaluation results (semi-annually)",
            ],
            "update_process": [
                "1. Receive and review change request",
                "2. Impact analysis and priority determination",
                "3. Content modification and internal review",
                "4. SME verification",
                "5. Final approval and deployment",
                "6. Record change history",
            ],
            "responsible": {
                "content_owner": "Instructional Design Team",
                "sme": "Subject Matter Experts",
                "reviewer": "Quality Assurance Team",
                "approver": "Training Operations Manager",
            },
        },
        "technical_maintenance": {
            "platform_updates": "Monthly regular inspection and updates",
            "backup_policy": "Daily automatic backup, weekly full backup",
            "security_review": "Quarterly security inspection",
            "performance_monitoring": "Real-time monitoring and monthly reports",
            "disaster_recovery": "Establish disaster recovery plan and annual DR drill",
        },
        "quality_assurance": {
            "review_checklist": [
                "Learning objective alignment verification",
                "Content accuracy verification",
                "Accessibility standards compliance",
                "User experience (UX) inspection",
                "Technical error testing",
            ],
            "feedback_channels": [
                "Post-learning satisfaction survey",
                "Real-time feedback button",
                "Regular FGI (Focus Group Interview)",
                "Line manager feedback",
            ],
            "improvement_cycle": "Collect feedback → Analysis → Derive improvements → Apply → Measure effectiveness",
        },
        "version_control": {
            "naming_convention": "v[Major].[Minor].[Patch]_YYYYMMDD",
            "major_version": "Curriculum structure change, learning objective change",
            "minor_version": "Content addition/modification, feature improvement",
            "patch_version": "Typo correction, bug fix",
            "changelog_management": "Record all changes and share with stakeholders",
            "archive_policy": "Retain previous versions for 2 years",
        },
        "support_resources": {
            "helpdesk": "Weekdays 09:00-18:00 operation",
            "self_service": "FAQ, user guide, video tutorials",
            "escalation_path": "1st Helpdesk → 2nd Technical Support Team → 3rd Development Team",
        },
    }
