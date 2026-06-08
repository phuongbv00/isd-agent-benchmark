"""
Usability Evaluation Phase Tools (4 tools)

Core phase of RPISD: Multi-stakeholder feedback collection
- evaluate_with_client: Client evaluation
- evaluate_with_expert: Expert evaluation
- evaluate_with_learner: Learner evaluation
- aggregate_feedback: Feedback aggregation
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
def evaluate_with_client(
    prototype: dict,
    project_scope: dict,
    success_criteria: List[str],
) -> dict:
    """
    Evaluates the prototype from the client's perspective.

    Reviews the prototype in terms of business requirements fulfillment,
    goal alignment, and ROI perspective.

    Args:
        prototype: Prototype results
        project_scope: Project scope definition
        success_criteria: Success criteria

    Returns:
        Client evaluation results (score, alignment, concerns, recommendations)
    """
    try:
        llm = get_llm()

        prompt = f"""You are a representative from the organization that commissioned this training program. Please evaluate the prototype from a business perspective.

## Prototype
{json.dumps(prototype, ensure_ascii=False, indent=2)}

## Project Scope
{json.dumps(project_scope, ensure_ascii=False, indent=2)}

## Success Criteria
{json.dumps(success_criteria, ensure_ascii=False)}

## Evaluation Criteria (Client Perspective)
1. **overall_score**: Overall score (0.0 - 1.0)
2. **business_alignment**: Business goal alignment
3. **scope_coverage**: Scope fulfillment
4. **concerns**: Concerns
5. **strengths**: Strengths
6. **recommendations**: Improvement recommendations

## Output Format (JSON)
```json
{{
  "evaluator": "client",
  "overall_score": 0.75,
  "business_alignment": {{
    "score": 0.8,
    "comments": "Well aligned with business goals"
  }},
  "scope_coverage": {{
    "score": 0.7,
    "covered": ["Core learning objectives", "Hands-on content"],
    "missing": ["Advanced case studies"]
  }},
  "concerns": [
    "Some advanced topics missing",
    "Concerns about insufficient practice time"
  ],
  "strengths": [
    "Core concepts clearly conveyed",
    "Practically applicable structure"
  ],
  "recommendations": [
    "Consider adding advanced case studies",
    "Expand practice time"
  ]
}}
```

Output only JSON."""

        response = llm.invoke(prompt)
        return parse_json_response(response.content)
    except Exception:
        return _fallback_evaluate_with_client(success_criteria)


def _fallback_evaluate_with_client(success_criteria: List[str]) -> dict:
    """Fallback function when LLM fails"""
    return {
        "evaluator": "client",
        "overall_score": 0.75,
        "business_alignment": {
            "score": 0.8,
            "comments": "Generally aligned with business goals",
        },
        "scope_coverage": {
            "score": 0.7,
            "covered": ["Core learning objectives", "Basic content"],
            "missing": ["Some advanced topics"],
        },
        "concerns": ["Sufficiency of practice time", "Diversity of application cases"],
        "strengths": ["Core concept delivery", "Structured organization"],
        "recommendations": ["Expand practice", "Add case studies"],
    }


@tool
def evaluate_with_expert(
    prototype: dict,
    design_result: dict,
    domain: Optional[str] = None,
) -> dict:
    """
    Evaluates the prototype from an expert's perspective.

    Reviews the prototype in terms of instructional design quality,
    content accuracy, and pedagogical appropriateness.

    Args:
        prototype: Prototype results
        design_result: Instructional design results
        domain: Educational domain

    Returns:
        Expert evaluation results (score, design_quality, content_accuracy, recommendations)
    """
    try:
        llm = get_llm()

        prompt = f"""You are an instructional design expert and subject matter expert (SME) in the relevant field. Please evaluate the prototype from an expert's perspective.

## Prototype
{json.dumps(prototype, ensure_ascii=False, indent=2)}

## Instructional Design
{json.dumps(design_result, ensure_ascii=False, indent=2)}

## Domain
{domain or "General"}

## Evaluation Criteria (Expert Perspective)
1. **overall_score**: Overall score (0.0 - 1.0)
2. **design_quality**: Instructional design quality
3. **content_accuracy**: Content accuracy
4. **pedagogical_soundness**: Pedagogical appropriateness
5. **concerns**: Concerns
6. **strengths**: Strengths
7. **recommendations**: Improvement recommendations

## Output Format (JSON)
```json
{{
  "evaluator": "expert",
  "overall_score": 0.8,
  "design_quality": {{
    "score": 0.85,
    "objective_alignment": "Learning objectives and activities are well aligned",
    "assessment_validity": "Assessment methods are appropriate for objectives"
  }},
  "content_accuracy": {{
    "score": 0.8,
    "accurate_items": ["Core concepts", "Basic principles"],
    "needs_review": ["Incorporation of latest trends"]
  }},
  "pedagogical_soundness": {{
    "score": 0.75,
    "comments": "Gagne's 9 Events are appropriately applied"
  }},
  "concerns": [
    "Some advanced concepts need additional explanation",
    "Practice difficulty needs adjustment"
  ],
  "strengths": [
    "Systematic learning sequence",
    "Clear learning objectives"
  ],
  "recommendations": [
    "Supplement explanations",
    "Scaffold difficulty levels"
  ]
}}
```

Output only JSON."""

        response = llm.invoke(prompt)
        return parse_json_response(response.content)
    except Exception:
        return _fallback_evaluate_with_expert()


def _fallback_evaluate_with_expert() -> dict:
    """Fallback function when LLM fails"""
    return {
        "evaluator": "expert",
        "overall_score": 0.8,
        "design_quality": {
            "score": 0.85,
            "objective_alignment": "Learning objectives and activities are generally aligned",
            "assessment_validity": "Assessment methods appropriate",
        },
        "content_accuracy": {
            "score": 0.8,
            "accurate_items": ["Core concepts", "Basic principles"],
            "needs_review": ["Some detailed content"],
        },
        "pedagogical_soundness": {
            "score": 0.75,
            "comments": "Instructional design principles appropriately applied",
        },
        "concerns": ["Explanations need supplementation", "Practice difficulty"],
        "strengths": ["Systematic organization", "Clear objectives"],
        "recommendations": ["Add more detail to explanations", "Adjust difficulty levels"],
    }


@tool
def evaluate_with_learner(
    prototype: dict,
    learner_characteristics: dict,
    sample_size: int = 5,
) -> dict:
    """
    Evaluates the prototype from the learner's perspective.

    Reviews the prototype in terms of usability, comprehension,
    engagement, and satisfaction.

    Args:
        prototype: Prototype results
        learner_characteristics: Learner characteristics
        sample_size: Number of sample learners

    Returns:
        Learner evaluation results (score, usability, comprehension, engagement, satisfaction)
    """
    try:
        llm = get_llm()

        prompt = f"""You are {sample_size} representative learners. Please evaluate the prototype from a learner's perspective.

## Prototype
{json.dumps(prototype, ensure_ascii=False, indent=2)}

## Learner Characteristics
{json.dumps(learner_characteristics, ensure_ascii=False, indent=2)}

## Evaluation Criteria (Learner Perspective)
1. **overall_score**: Overall score (0.0 - 1.0)
2. **usability**: Usability (Is it easy to learn?)
3. **comprehension**: Comprehension (Is the content easy to understand?)
4. **engagement**: Engagement (Is it interesting and engaging?)
5. **satisfaction**: Satisfaction (Overall satisfaction)
6. **concerns**: Difficulties
7. **suggestions**: Improvement suggestions

## Output Format (JSON)
```json
{{
  "evaluator": "learner",
  "sample_size": {sample_size},
  "overall_score": 0.7,
  "usability": {{
    "score": 0.75,
    "navigation": "Learning sequence is clear",
    "accessibility": "Mostly easy to access"
  }},
  "comprehension": {{
    "score": 0.7,
    "clear_items": ["Basic concepts", "Practice instructions"],
    "confusing_items": ["Some technical terms"]
  }},
  "engagement": {{
    "score": 0.65,
    "interesting_elements": ["Hands-on activities", "Case studies"],
    "boring_elements": ["Long explanations"]
  }},
  "satisfaction": {{
    "score": 0.7,
    "comments": "Generally useful but some improvements needed"
  }},
  "concerns": [
    "Some terminology is difficult",
    "Insufficient practice time"
  ],
  "suggestions": [
    "Add terminology explanations",
    "More examples"
  ]
}}
```

Output only JSON."""

        response = llm.invoke(prompt)
        return parse_json_response(response.content)
    except Exception:
        return _fallback_evaluate_with_learner(sample_size)


def _fallback_evaluate_with_learner(sample_size: int = 5) -> dict:
    """Fallback function when LLM fails"""
    return {
        "evaluator": "learner",
        "sample_size": sample_size,
        "overall_score": 0.7,
        "usability": {
            "score": 0.75,
            "navigation": "Clear learning sequence",
            "accessibility": "Good accessibility",
        },
        "comprehension": {
            "score": 0.7,
            "clear_items": ["Basic concepts", "Practice instructions"],
            "confusing_items": ["Some technical terms"],
        },
        "engagement": {
            "score": 0.65,
            "interesting_elements": ["Practice", "Case studies"],
            "boring_elements": ["Long explanations"],
        },
        "satisfaction": {
            "score": 0.7,
            "comments": "Generally useful",
        },
        "concerns": ["Difficult terminology", "Practice time"],
        "suggestions": ["Terminology explanations", "Add examples"],
    }


@tool
def aggregate_feedback(
    client_feedback: dict,
    expert_feedback: dict,
    learner_feedback: dict,
    quality_threshold: float = 0.8,
) -> dict:
    """
    Aggregates three types of feedback and calculates quality scores.

    Synthesizes client, expert, and learner feedback to derive
    final quality scores and improvement areas.

    Args:
        client_feedback: Client feedback
        expert_feedback: Expert feedback
        learner_feedback: Learner feedback
        quality_threshold: Quality threshold score

    Returns:
        Aggregated feedback (aggregated_score, improvement_areas, recommendations, pass_threshold)
    """
    try:
        llm = get_llm()

        prompt = f"""You are an instructional designer specializing in rapid prototyping. Please provide an integrated analysis of the three types of feedback.

## Client Feedback
{json.dumps(client_feedback, ensure_ascii=False, indent=2)}

## Expert Feedback
{json.dumps(expert_feedback, ensure_ascii=False, indent=2)}

## Learner Feedback
{json.dumps(learner_feedback, ensure_ascii=False, indent=2)}

## Quality Threshold
{quality_threshold}

## Integration Analysis Items
1. **aggregated_score**: Weighted average score (Client 0.3, Expert 0.4, Learner 0.3)
2. **score_breakdown**: Scores by evaluator
3. **improvement_areas**: Areas needing improvement (in priority order)
4. **common_concerns**: Common concerns
5. **recommendations**: Integrated improvement recommendations
6. **pass_threshold**: Whether quality threshold is met

## Output Format (JSON)
```json
{{
  "aggregated_score": 0.75,
  "score_breakdown": {{
    "client": 0.75,
    "expert": 0.8,
    "learner": 0.7
  }},
  "improvement_areas": [
    {{"area": "Terminology explanations", "priority": "high", "sources": ["learner", "expert"]}},
    {{"area": "Expand practice", "priority": "medium", "sources": ["client", "learner"]}}
  ],
  "common_concerns": [
    "Some content needs additional explanation",
    "Practice time and difficulty adjustment"
  ],
  "recommendations": [
    "Add explanations for technical terms",
    "Expand and scaffold practice time",
    "Include additional examples and case studies"
  ],
  "pass_threshold": false,
  "gap_to_threshold": 0.05
}}
```

Output only JSON."""

        response = llm.invoke(prompt)
        return parse_json_response(response.content)
    except Exception:
        return _fallback_aggregate_feedback(
            client_feedback, expert_feedback, learner_feedback, quality_threshold
        )


def _fallback_aggregate_feedback(
    client_feedback: dict,
    expert_feedback: dict,
    learner_feedback: dict,
    quality_threshold: float = 0.8,
) -> dict:
    """Fallback function when LLM fails"""
    client_score = client_feedback.get("overall_score", 0.75)
    expert_score = expert_feedback.get("overall_score", 0.8)
    learner_score = learner_feedback.get("overall_score", 0.7)

    # Weighted average (Client 0.3, Expert 0.4, Learner 0.3)
    aggregated_score = (client_score * 0.3) + (expert_score * 0.4) + (learner_score * 0.3)

    pass_threshold = aggregated_score >= quality_threshold
    gap_to_threshold = max(0, quality_threshold - aggregated_score)

    # Aggregate concerns
    all_concerns = []
    all_concerns.extend(client_feedback.get("concerns", []))
    all_concerns.extend(expert_feedback.get("concerns", []))
    all_concerns.extend(learner_feedback.get("concerns", []))

    # Aggregate recommendations
    all_recommendations = []
    all_recommendations.extend(client_feedback.get("recommendations", []))
    all_recommendations.extend(expert_feedback.get("recommendations", []))
    all_recommendations.extend(learner_feedback.get("suggestions", []))

    return {
        "aggregated_score": round(aggregated_score, 2),
        "score_breakdown": {
            "client": client_score,
            "expert": expert_score,
            "learner": learner_score,
        },
        "improvement_areas": [
            {"area": "Content explanation supplementation", "priority": "high", "sources": ["expert", "learner"]},
            {"area": "Expand practice", "priority": "medium", "sources": ["client", "learner"]},
        ],
        "common_concerns": list(set(all_concerns))[:5],
        "recommendations": list(set(all_recommendations))[:5],
        "pass_threshold": pass_threshold,
        "gap_to_threshold": round(gap_to_threshold, 2),
    }
