"""Deterministic candidate scoring for the Redrob Senior AI Engineer role."""

from __future__ import annotations

from typing import Any

from features import extract_features
from preprocessing import normalize_text
from risk_checks import compute_risk_penalty


RELEVANT_TITLE_KEYWORDS = [
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

PRODUCT_INDUSTRY_KEYWORDS = [
    "software",
    "saas",
    "product",
    "platform",
    "startup",
    "marketplace",
    "internet",
    "fintech",
    "e-commerce",
    "ai-native",
]

PREFERRED_LOCATION_KEYWORDS = [
    "pune",
    "noida",
    "delhi",
    "ncr",
    "gurgaon",
    "gurugram",
    "hyderabad",
    "bangalore",
    "bengaluru",
    "mumbai",
]


def clamp(value: float, low: float, high: float) -> float:
    """Clamp a value into an inclusive numeric range."""

    return max(low, min(high, value))


def _safe_number(value: Any, default: float = 0.0) -> float:
    """Convert simple numeric values to float without raising."""

    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return default
    return default


def _is_true(value: Any) -> bool:
    """Return True only for explicit true-like values."""

    return value is True


def _contains_any(text: str, keywords: list[str]) -> bool:
    """Return whether normalized text contains any configured keyword."""

    normalized = normalize_text(text)
    return any(keyword in normalized for keyword in keywords)


def is_relevant_title(title: Any) -> bool:
    """Return whether the current title looks relevant to the JD."""

    return _contains_any(str(title or ""), RELEVANT_TITLE_KEYWORDS)


def _technical_fit(features: dict[str, Any]) -> float:
    """Score technical fit, weighting career evidence over skills."""

    career_retrieval = _safe_number(features.get("career_retrieval_keyword_count"))
    skill_retrieval = _safe_number(features.get("skill_retrieval_keyword_count"))
    career_vector = _safe_number(features.get("career_vector_keyword_count"))
    skill_vector = _safe_number(features.get("skill_vector_keyword_count"))
    career_llm = _safe_number(features.get("career_llm_nlp_keyword_count"))
    skill_llm = _safe_number(features.get("skill_llm_nlp_keyword_count"))
    career_eval = _safe_number(features.get("career_evaluation_keyword_count"))
    skill_eval = _safe_number(features.get("skill_evaluation_keyword_count"))
    career_prod = _safe_number(features.get("career_production_keyword_count"))
    skill_prod = _safe_number(features.get("skill_production_keyword_count"))

    score = 0.0
    score += min(career_retrieval * 3.0 + skill_retrieval * 0.8, 8.0)
    score += min(career_vector * 3.0 + skill_vector * 0.8, 7.0)
    score += min(career_llm * 2.0 + skill_llm * 0.6, 5.0)
    score += min(career_eval * 2.5 + skill_eval * 0.6, 5.0)
    score += min(career_prod * 1.8 + skill_prod * 0.5, 5.0)

    if features.get("has_recommendation_evidence"):
        score += 2.0
    if features.get("has_finetuning_evidence"):
        score += 1.0
    if features.get("has_embedding_evidence"):
        score += 1.5
    if features.get("has_python_evidence"):
        score += 0.5

    return round(clamp(score, 0.0, 35.0), 4)


def _career_evidence(features: dict[str, Any]) -> float:
    """Score evidence from real roles, profile text, and current title."""

    score = 0.0
    score += min(_safe_number(features.get("career_retrieval_keyword_count")) * 4.0, 8.0)
    score += min(_safe_number(features.get("career_vector_keyword_count")) * 4.0, 6.0)
    score += min(_safe_number(features.get("career_evaluation_keyword_count")) * 3.0, 4.0)
    score += min(_safe_number(features.get("career_production_keyword_count")) * 2.0, 4.0)

    if is_relevant_title(features.get("current_title")):
        score += 3.0
    if features.get("has_ranking_evidence"):
        score += 1.5
    if features.get("has_recommendation_evidence"):
        score += 1.5

    role_count = _safe_number(features.get("career_role_count"))
    if role_count >= 2:
        score += 1.0

    return round(clamp(score, 0.0, 25.0), 4)


def _experience_fit(features: dict[str, Any]) -> float:
    """Score years of experience against the preferred 5-9 year band."""

    years = _safe_number(features.get("years_of_experience"), default=-1.0)
    if 5 <= years <= 9:
        return 12.0
    if 4 <= years <= 10:
        return 10.0
    if 3 <= years <= 12:
        return 7.0
    if years > 12:
        return 4.0
    if 0 <= years < 3:
        return 3.0
    return 0.0


def _product_company_fit(features: dict[str, Any]) -> float:
    """Score product/SaaS/startup background and discount consulting-only history."""

    score = 0.0
    if features.get("has_product_company_signal"):
        score += 6.0

    industry_text = " ".join(
        [
            str(features.get("current_company") or ""),
            str(features.get("current_industry") or ""),
            str(features.get("career_text") or ""),
        ]
    )
    if _contains_any(industry_text, PRODUCT_INDUSTRY_KEYWORDS):
        score += 3.0

    if features.get("consulting_only_background"):
        score -= 4.0
    elif features.get("has_consulting_service_company"):
        score -= 1.0

    return round(clamp(score, 0.0, 10.0), 4)


def _behavioral_fit(features: dict[str, Any]) -> float:
    """Score reachability, activity, and hiring-process reliability."""

    score = 0.0
    if _is_true(features.get("open_to_work_flag")):
        score += 1.5

    response_rate = _safe_number(features.get("recruiter_response_rate"), default=-1.0)
    if response_rate >= 0.70:
        score += 1.5
    elif response_rate >= 0.40:
        score += 1.0
    elif response_rate >= 0.20:
        score += 0.5

    response_hours = _safe_number(features.get("avg_response_time_hours"), default=-1.0)
    if 0 <= response_hours <= 24:
        score += 1.0
    elif 24 < response_hours <= 72:
        score += 0.5

    notice_days = _safe_number(features.get("notice_period_days"), default=999.0)
    if notice_days <= 30:
        score += 1.0
    elif notice_days <= 60:
        score += 0.7
    elif notice_days <= 90:
        score += 0.3

    saved_count = _safe_number(features.get("saved_by_recruiters_30d"))
    if saved_count >= 5:
        score += 1.0
    elif saved_count >= 1:
        score += 0.5

    interview_rate = _safe_number(features.get("interview_completion_rate"), default=-1.0)
    if interview_rate >= 0.80:
        score += 1.0
    elif interview_rate >= 0.50:
        score += 0.5

    offer_rate = _safe_number(features.get("offer_acceptance_rate"), default=-1.0)
    if offer_rate >= 0.70:
        score += 0.8
    elif offer_rate >= 0.40:
        score += 0.4

    if _is_true(features.get("verified_email")):
        score += 0.5
    if _is_true(features.get("verified_phone")):
        score += 0.5
    if _is_true(features.get("linkedin_connected")):
        score += 0.5

    completeness = _safe_number(features.get("profile_completeness_score"))
    if completeness >= 90:
        score += 1.2
    elif completeness >= 70:
        score += 0.8
    elif completeness >= 50:
        score += 0.4

    return round(clamp(score, 0.0, 10.0), 4)


def _location_fit(features: dict[str, Any]) -> float:
    """Score location preference and relocation flexibility."""

    location_text = normalize_text(
        " ".join([str(features.get("location") or ""), str(features.get("country") or "")])
    )
    score = 0.0
    if _contains_any(location_text, PREFERRED_LOCATION_KEYWORDS):
        score += 4.0
    elif "india" in location_text:
        score += 3.0

    if _is_true(features.get("willing_to_relocate")):
        score += 1.0

    return round(clamp(score, 0.0, 5.0), 4)


def _github_fit(features: dict[str, Any]) -> float:
    """Score GitHub activity when available."""

    github_score = _safe_number(features.get("github_activity_score"), default=-1.0)
    if github_score > 70:
        return 3.0
    if github_score > 40:
        return 2.2
    if github_score > 15:
        return 1.4
    if github_score >= 0:
        return 0.5
    return 0.0


def score_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """Score one candidate and return components, risk, and features."""

    features = extract_features(candidate)
    risk_penalty, risk_reasons = compute_risk_penalty(candidate, features)

    technical_fit = _technical_fit(features)
    career_evidence = _career_evidence(features)
    experience_fit = _experience_fit(features)
    product_company_fit = _product_company_fit(features)
    behavioral_fit = _behavioral_fit(features)
    location_fit = _location_fit(features)
    github_fit = _github_fit(features)

    base_score = round(
        technical_fit
        + career_evidence
        + experience_fit
        + product_company_fit
        + behavioral_fit
        + location_fit
        + github_fit,
        4,
    )
    final_score = round(clamp(base_score - (risk_penalty * 35.0), 0.0, 100.0), 4)

    return {
        "candidate_id": str(features.get("candidate_id") or ""),
        "score": final_score,
        "base_score": base_score,
        "technical_fit": technical_fit,
        "career_evidence": career_evidence,
        "experience_fit": experience_fit,
        "product_company_fit": product_company_fit,
        "behavioral_fit": behavioral_fit,
        "location_fit": location_fit,
        "github_fit": github_fit,
        "risk_penalty": risk_penalty,
        "risk_reasons": risk_reasons,
        "features": features,
    }


def score_placeholder(features: dict[str, float]) -> float:
    """Return a simple deterministic starter score."""

    keyword_score = features.get("keyword_hits", 0.0) * 10.0
    length_score = min(features.get("text_length", 0.0) / 1000.0, 5.0)
    return round(keyword_score + length_score, 4)


if __name__ == "__main__":
    strong_candidate = {
        "candidate_id": "strong-search-ai",
        "profile": {
            "headline": "Senior AI Engineer - Search and Ranking",
            "summary": "Built semantic search, recommendations, RAG, and ranking platforms.",
            "location": "Bengaluru",
            "country": "India",
            "years_of_experience": 7,
            "current_title": "Senior AI Engineer",
            "current_company": "Search Platform AI",
            "current_company_size": "51-200",
            "current_industry": "SaaS product",
        },
        "career_history": [
            {
                "title": "Senior AI Engineer",
                "company": "Search Platform AI",
                "industry": "Software product",
                "company_size": "51-200",
                "duration_months": 36,
                "is_current": True,
                "description": (
                    "Owned BM25, hybrid search, vector database retrieval, embeddings, "
                    "Python FastAPI deployment, NDCG/MRR evaluation, and A/B testing."
                ),
            },
            {
                "title": "Machine Learning Engineer",
                "company": "Marketplace Startup",
                "industry": "Marketplace",
                "description": "Built recommendation systems and model serving pipelines.",
            },
        ],
        "skills": [
            {"name": "Python", "proficiency": "expert", "duration_months": 84},
            {"name": "vector db", "proficiency": "advanced", "duration_months": 36},
            {"name": "RAG", "proficiency": "advanced", "duration_months": 24},
            {"name": "NDCG", "proficiency": "advanced", "duration_months": 24},
        ],
        "education": [{"institution": "Example University", "degree": "MS"}],
        "redrob_signals": {
            "signup_date": "2024-01-15",
            "last_active_date": "2026-06-01",
            "expected_salary_range_inr_lpa": {"min": 45, "max": 70},
            "profile_completeness_score": 95,
            "open_to_work_flag": True,
            "recruiter_response_rate": 0.82,
            "avg_response_time_hours": 12,
            "notice_period_days": 30,
            "saved_by_recruiters_30d": 8,
            "interview_completion_rate": 0.90,
            "offer_acceptance_rate": 0.75,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
            "willing_to_relocate": True,
            "github_activity_score": 78,
        },
    }

    weak_candidate = {
        "candidate_id": "weak-marketing",
        "profile": {
            "headline": "Marketing Manager",
            "summary": "Business and campaign management profile.",
            "location": "London",
            "country": "United Kingdom",
            "years_of_experience": 2,
            "current_title": "Marketing Manager",
            "current_company": "TCS",
            "current_industry": "Consulting",
        },
        "career_history": [
            {
                "title": "Marketing Manager",
                "company": "TCS",
                "industry": "Consulting",
                "description": "Managed brand campaigns and events.",
            }
        ],
        "skills": [
            {"name": "vector db", "proficiency": "expert", "duration_months": 0},
            {"name": "FAISS", "proficiency": "expert", "duration_months": 0},
            {"name": "Qdrant", "proficiency": "expert", "duration_months": 0},
            {"name": "RAG", "proficiency": "advanced", "duration_months": 0},
            {"name": "LLM", "proficiency": "advanced", "duration_months": 0},
            {"name": "NDCG", "proficiency": "advanced", "duration_months": 0},
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
            "notice_period_days": 120,
        },
    }

    strong_score = score_candidate(strong_candidate)
    weak_score = score_candidate(weak_candidate)
    assert strong_score["score"] > weak_score["score"]
    assert strong_score["score"] > 60
    assert weak_score["score"] < 50
    print(
        {
            "strong": {
                "score": strong_score["score"],
                "base_score": strong_score["base_score"],
                "technical_fit": strong_score["technical_fit"],
                "career_evidence": strong_score["career_evidence"],
                "risk_penalty": strong_score["risk_penalty"],
            },
            "weak": {
                "score": weak_score["score"],
                "base_score": weak_score["base_score"],
                "technical_fit": weak_score["technical_fit"],
                "career_evidence": weak_score["career_evidence"],
                "risk_penalty": weak_score["risk_penalty"],
                "risk_reasons": weak_score["risk_reasons"],
            },
        }
    )
