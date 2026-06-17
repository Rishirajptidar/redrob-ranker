"""Risk checks and lightweight candidate validation helpers."""

from __future__ import annotations

from typing import Any

from features import extract_features
from preprocessing import normalize_text


TITLE_FIT_KEYWORDS = [
    "ai",
    "ml",
    "machine learning",
    "data scientist",
    "data engineer",
    "software engineer",
    "backend",
    "nlp",
    "search",
    "ranking",
    "recommendation",
    "platform",
    "engineer",
]

NON_IR_SPECIALIZATION_KEYWORDS = [
    "computer vision",
    "object detection",
    "image classification",
    "opencv",
    "cnn",
    "gan",
    "speech recognition",
    "tts",
    "robotics",
]


def has_candidate_id(candidate: dict[str, Any]) -> bool:
    """Return whether a candidate has a usable identifier field."""

    return get_candidate_id(candidate) != ""


def get_candidate_id(candidate: dict[str, Any]) -> str:
    """Read the candidate id from common identifier fields."""

    raw_id = candidate.get("candidate_id", candidate.get("id", ""))
    return str(raw_id).strip()


def _as_number(value: Any) -> int | float | None:
    """Return numeric values while ignoring booleans and invalid strings."""

    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        try:
            parsed = float(value.strip())
        except ValueError:
            return None
        return int(parsed) if parsed.is_integer() else parsed
    return None


def _contains_any(text: str, keywords: list[str]) -> bool:
    """Return whether normalized text contains any keyword."""

    normalized = normalize_text(text)
    return any(keyword in normalized for keyword in keywords)


def _keyword_count(text: str, keywords: list[str]) -> int:
    """Count distinct specialization keywords present in text."""

    normalized = normalize_text(text)
    return sum(1 for keyword in keywords if keyword in normalized)


def _add_penalty(
    reasons: list[str],
    reason: str,
    current_penalty: float,
    amount: float,
) -> float:
    """Append a machine-readable reason and add its penalty."""

    reasons.append(reason)
    return current_penalty + amount


def compute_risk_penalty(
    candidate: dict[str, Any], features: dict[str, Any] | None = None
) -> tuple[float, list[str]]:
    """Compute an additive risk penalty and machine-readable reasons."""

    feature_values = extract_features(candidate) if features is None else features
    reasons: list[str] = []
    penalty = 0.0

    salary_min = _as_number(feature_values.get("salary_min"))
    salary_max = _as_number(feature_values.get("salary_max"))
    signup_date = feature_values.get("parsed_signup_date")
    last_active_date = feature_values.get("parsed_last_active_date")
    career_retrieval = _as_number(feature_values.get("career_retrieval_keyword_count")) or 0
    career_vector = _as_number(feature_values.get("career_vector_keyword_count")) or 0
    career_evaluation = _as_number(feature_values.get("career_evaluation_keyword_count")) or 0
    career_production = _as_number(feature_values.get("career_production_keyword_count")) or 0
    career_llm_nlp = _as_number(feature_values.get("career_llm_nlp_keyword_count")) or 0
    advanced_target_skills = _as_number(feature_values.get("advanced_target_skill_count")) or 0
    expert_skills = _as_number(feature_values.get("expert_skill_count")) or 0
    zero_duration_expert_skills = (
        _as_number(feature_values.get("zero_duration_expert_skill_count")) or 0
    )
    notice_period_days = _as_number(feature_values.get("notice_period_days"))
    recruiter_response_rate = _as_number(feature_values.get("recruiter_response_rate"))
    interview_completion_rate = _as_number(feature_values.get("interview_completion_rate"))
    github_activity_score = _as_number(feature_values.get("github_activity_score"))
    target_skill_count = _as_number(feature_values.get("target_skill_count")) or 0

    if salary_min is not None and salary_max is not None and salary_min > salary_max:
        penalty = _add_penalty(reasons, "salary_min_greater_than_max", penalty, 0.20)

    if signup_date is not None and last_active_date is not None and signup_date > last_active_date:
        penalty = _add_penalty(reasons, "signup_after_last_active", penalty, 0.25)

    if (
        advanced_target_skills >= 6
        and career_retrieval + career_vector + career_evaluation == 0
    ):
        penalty = _add_penalty(
            reasons,
            "skill_keyword_stuffing_without_career_evidence",
            penalty,
            0.25,
        )

    if expert_skills >= 8 and career_retrieval + career_vector <= 1:
        penalty = _add_penalty(
            reasons,
            "too_many_expert_skills_weak_career_support",
            penalty,
            0.20,
        )

    if zero_duration_expert_skills >= 2:
        penalty = _add_penalty(
            reasons,
            "expert_skills_with_zero_duration",
            penalty,
            0.20,
        )

    current_title = normalize_text(feature_values.get("current_title"))
    if (
        current_title
        and not _contains_any(current_title, TITLE_FIT_KEYWORDS)
        and advanced_target_skills >= 5
    ):
        penalty = _add_penalty(
            reasons,
            "irrelevant_title_with_ai_keywords",
            penalty,
            0.25,
        )

    if feature_values.get("consulting_only_background") is True:
        penalty = _add_penalty(reasons, "consulting_only_background", penalty, 0.15)

    if notice_period_days is not None and notice_period_days > 90:
        penalty = _add_penalty(reasons, "long_notice_period", penalty, 0.12)

    if recruiter_response_rate is not None and recruiter_response_rate < 0.15:
        penalty = _add_penalty(reasons, "low_recruiter_response_rate", penalty, 0.10)

    if interview_completion_rate is not None and interview_completion_rate < 0.40:
        penalty = _add_penalty(reasons, "low_interview_completion_rate", penalty, 0.10)

    if (
        github_activity_score == -1
        and career_retrieval + career_vector + career_production <= 1
    ):
        penalty = _add_penalty(
            reasons,
            "no_github_and_weak_technical_evidence",
            penalty,
            0.12,
        )

    if (
        feature_values.get("open_to_work_flag") is False
        and recruiter_response_rate is not None
        and recruiter_response_rate < 0.25
    ):
        penalty = _add_penalty(reasons, "poor_availability_signal", penalty, 0.10)

    if (
        _keyword_count(feature_values.get("all_text", ""), NON_IR_SPECIALIZATION_KEYWORDS) >= 3
        and career_retrieval == 0
        and career_llm_nlp == 0
    ):
        penalty = _add_penalty(
            reasons,
            "non_ir_ai_specialization_mismatch",
            penalty,
            0.15,
        )

    if (
        career_retrieval + career_vector + career_evaluation + career_production == 0
        and target_skill_count <= 2
    ):
        penalty = _add_penalty(reasons, "very_weak_jd_fit", penalty, 0.25)

    return min(round(penalty, 4), 1.0), reasons


if __name__ == "__main__":
    suspicious_candidate = {
        "candidate_id": "suspicious-1",
        "profile": {
            "headline": "Marketing Manager",
            "summary": "Broad business leadership profile.",
            "current_title": "Marketing Manager",
            "current_company": "Example Services",
            "current_industry": "Consulting",
        },
        "career_history": [
            {
                "title": "Marketing Manager",
                "company": "TCS",
                "industry": "Consulting",
                "description": "Managed campaigns and brand operations.",
            }
        ],
        "skills": [
            {"name": "vector db", "proficiency": "advanced", "duration_months": 0},
            {"name": "FAISS", "proficiency": "expert", "duration_months": 0},
            {"name": "Qdrant", "proficiency": "expert", "duration_months": 0},
            {"name": "RAG", "proficiency": "advanced", "duration_months": 0},
            {"name": "LLM", "proficiency": "expert", "duration_months": 0},
            {"name": "NDCG", "proficiency": "advanced", "duration_months": 0},
            {"name": "MLOps", "proficiency": "advanced", "duration_months": 0},
        ],
        "education": [],
        "redrob_signals": {
            "signup_date": "2025-02-01",
            "last_active_date": "2025-01-01",
            "expected_salary_range_inr_lpa": {"min": 80, "max": 40},
            "github_activity_score": -1,
            "recruiter_response_rate": 0.10,
            "interview_completion_rate": 0.20,
            "open_to_work_flag": False,
        },
    }

    penalty, reasons = compute_risk_penalty(suspicious_candidate)
    assert penalty > 0.5
    assert "salary_min_greater_than_max" in reasons
    assert "signup_after_last_active" in reasons
    assert (
        "skill_keyword_stuffing_without_career_evidence" in reasons
        or "irrelevant_title_with_ai_keywords" in reasons
    )
    print({"penalty": penalty, "reasons": reasons})
