"""Grounded, concise reasoning strings for ranked candidates."""

from __future__ import annotations

from typing import Any

from scoring import score_candidate


RISK_CONCERN_LABELS = {
    "salary_min_greater_than_max": "salary data inconsistency",
    "signup_after_last_active": "activity date inconsistency",
    "skill_keyword_stuffing_without_career_evidence": "skills are stronger than career evidence",
    "irrelevant_title_with_ai_keywords": "title does not align strongly with AI engineering",
    "consulting_only_background": "mostly consulting/services background",
    "long_notice_period": "long notice period",
    "low_recruiter_response_rate": "low recruiter response rate",
    "low_interview_completion_rate": "low interview completion signal",
    "no_github_and_weak_technical_evidence": "limited external technical signal",
}


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


def _fit_label(score: float) -> str:
    """Return a compact qualitative fit label."""

    if score >= 75:
        return "Strong fit"
    if score >= 55:
        return "Good fit"
    return "Moderate fit"


def _title_experience_phrase(features: dict[str, Any]) -> str:
    """Build a grounded title and experience phrase."""

    title = str(features.get("current_title") or "").strip()
    years = _safe_number(features.get("years_of_experience"), default=-1.0)
    if years >= 0:
        years_text = str(int(years)) if years.is_integer() else f"{years:.1f}"
        if title:
            return f"{title} with {years_text} years"
        return f"{years_text} years of experience"
    if title:
        return title
    return "Candidate"


def _technical_phrases(features: dict[str, Any]) -> list[str]:
    """Build concise evidence phrases ordered by strength."""

    phrase_scores: list[tuple[float, str]] = []
    career_retrieval = _safe_number(features.get("career_retrieval_keyword_count"))
    skill_retrieval = _safe_number(features.get("skill_retrieval_keyword_count"))
    career_vector = _safe_number(features.get("career_vector_keyword_count"))
    skill_vector = _safe_number(features.get("skill_vector_keyword_count"))
    career_eval = _safe_number(features.get("career_evaluation_keyword_count"))
    skill_eval = _safe_number(features.get("skill_evaluation_keyword_count"))
    career_prod = _safe_number(features.get("career_production_keyword_count"))
    skill_prod = _safe_number(features.get("skill_production_keyword_count"))
    career_llm = _safe_number(features.get("career_llm_nlp_keyword_count"))
    skill_llm = _safe_number(features.get("skill_llm_nlp_keyword_count"))

    if career_retrieval > 0 or skill_retrieval > 0 or features.get("has_ranking_evidence"):
        phrase_scores.append(
            (
                career_retrieval * 2.5 + skill_retrieval,
                "retrieval/search/ranking evidence",
            )
        )
    if features.get("has_recommendation_evidence"):
        phrase_scores.append((3.0, "recommendation-system exposure"))
    if career_vector > 0 or skill_vector > 0 or features.get("has_vector_db_evidence"):
        phrase_scores.append(
            (career_vector * 2.3 + skill_vector, "vector search/embedding evidence")
        )
    if career_eval > 0 or skill_eval > 0 or features.get("has_evaluation_evidence"):
        phrase_scores.append(
            (career_eval * 2.0 + skill_eval, "ranking/evaluation signals")
        )
    if career_prod > 0 or skill_prod > 0 or features.get("has_backend_or_production_evidence"):
        phrase_scores.append(
            (career_prod * 1.8 + skill_prod, "Python/backend production work")
        )
    if career_llm > 0 or skill_llm > 0 or features.get("has_llm_nlp_evidence"):
        phrase_scores.append((career_llm * 1.7 + skill_llm, "LLM/NLP relevance"))
    if features.get("has_mlops_evidence"):
        phrase_scores.append((1.5, "MLOps/deployment signal"))

    phrase_scores.sort(key=lambda item: (-item[0], item[1]))
    return [phrase for _, phrase in phrase_scores[:4]]


def _context_phrases(features: dict[str, Any], scored: dict[str, Any]) -> list[str]:
    """Build short context phrases for experience, company, location, and behavior."""

    phrases: list[str] = []
    if features.get("has_product_company_signal"):
        phrases.append("product/SaaS background supports practical fit")

    if scored.get("location_fit", 0) or features.get("willing_to_relocate"):
        phrases.append("India/location or relocation signals help")

    notice_days = _safe_number(features.get("notice_period_days"), default=999.0)
    if features.get("open_to_work_flag") is True and notice_days <= 60:
        phrases.append("open-to-work and short notice improve reachability")
    elif features.get("open_to_work_flag") is True:
        phrases.append("open-to-work signal improves reachability")

    response_rate = _safe_number(features.get("recruiter_response_rate"), default=-1.0)
    if response_rate >= 0.4:
        phrases.append("strong response signals improve hireability")

    github_score = _safe_number(features.get("github_activity_score"), default=-1.0)
    if github_score > 40:
        phrases.append("active GitHub signal adds technical validation")

    return phrases


def _join_phrases(phrases: list[str], limit: int = 4) -> str:
    """Join a few phrases without making the sentence sprawl."""

    selected = phrases[:limit]
    if not selected:
        return "limited direct evidence for production retrieval/ranking"
    if len(selected) == 1:
        return selected[0]
    return ", ".join(selected[:-1]) + ", and " + selected[-1]


def _sentence_start(text: str) -> str:
    """Uppercase the first character while preserving the rest of the phrase."""

    if not text:
        return text
    return text[0].upper() + text[1:]


def _concern_text(scored: dict[str, Any]) -> str:
    """Return one concise concern sentence fragment when risk is meaningful."""

    risk_penalty = _safe_number(scored.get("risk_penalty"))
    if risk_penalty < 0.15:
        return ""

    reasons = scored.get("risk_reasons", [])
    if not isinstance(reasons, list):
        return ""

    concerns = [
        RISK_CONCERN_LABELS[reason]
        for reason in reasons
        if reason in RISK_CONCERN_LABELS
    ]
    if not concerns:
        return ""
    return f" Concern: {concerns[0]}."


def generate_reasoning(candidate: dict[str, Any], scored: dict[str, Any] | None = None) -> str:
    """Generate a grounded 1-2 sentence ranking explanation."""

    scored_candidate = score_candidate(candidate) if scored is None else scored
    features = scored_candidate.get("features", {})
    score = _safe_number(scored_candidate.get("score"))
    label = _fit_label(score)

    evidence = _technical_phrases(features)
    context = _context_phrases(features, scored_candidate)
    subject = _title_experience_phrase(features)
    if evidence:
        first_sentence = f"{label}: {subject} showing {_join_phrases(evidence)}."
    else:
        first_sentence = f"{label}: {subject}; {_join_phrases(context, limit=2)}."

    if evidence and context:
        second_sentence = f"{_sentence_start(_join_phrases(context, limit=2))}."
    elif score < 55:
        second_sentence = "Career evidence for production retrieval/ranking is limited."
    else:
        second_sentence = ""

    reasoning = " ".join(part for part in [first_sentence, second_sentence] if part)
    reasoning += _concern_text(scored_candidate)

    if len(reasoning) > 330 and " Also has " in reasoning:
        reasoning = first_sentence + _concern_text(scored_candidate)
    return reasoning[:349]


def build_placeholder_reasoning(features: dict[str, float]) -> str:
    """Create a concise explanation for the placeholder score."""

    hits = int(features.get("keyword_hits", 0.0))
    return f"Placeholder score based on {hits} senior AI keyword matches."


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
                "duration_months": 36,
                "is_current": True,
                "description": (
                    "Owned BM25, hybrid search, vector database retrieval, embeddings, "
                    "Python FastAPI deployment, NDCG/MRR evaluation, and A/B testing."
                ),
            }
        ],
        "skills": [
            {"name": "Python", "proficiency": "expert", "duration_months": 84},
            {"name": "vector db", "proficiency": "advanced", "duration_months": 36},
            {"name": "RAG", "proficiency": "advanced", "duration_months": 24},
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

    scored = score_candidate(strong_candidate)
    reasoning = generate_reasoning(strong_candidate, scored)
    assert reasoning
    assert "placeholder" not in reasoning.lower()
    assert len(reasoning) < 350
    print(reasoning)
