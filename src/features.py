"""Deterministic feature extraction for Senior AI Engineer candidate ranking.

All features are computed from one candidate dictionary at a time using only the
Python standard library. This keeps the pipeline CPU-only, deterministic, and
safe for streaming large JSONL files.
"""

from __future__ import annotations

from typing import Any

from preprocessing import normalize_skill_name, normalize_text, preprocess_candidate


RETRIEVAL_KEYWORDS = [
    "retrieval",
    "search",
    "ranking",
    "ranker",
    "reranking",
    "re-ranking",
    "semantic search",
    "hybrid search",
    "bm25",
    "information retrieval",
    "learning to rank",
]

RECOMMENDATION_KEYWORDS = [
    "recommendation",
    "recommender",
    "recommendation system",
    "personalization",
    "matching system",
]

VECTOR_KEYWORDS = [
    "embedding",
    "embeddings",
    "vector search",
    "vector database",
    "faiss",
    "pinecone",
    "qdrant",
    "weaviate",
    "milvus",
    "elastic search",
    "elasticsearch",
    "open search",
    "opensearch",
    "approximate nearest neighbor",
    "ann",
]

LLM_NLP_KEYWORDS = [
    "llm",
    "rag",
    "nlp",
    "transformer",
    "transformers",
    "hugging face",
    "prompt engineering",
    "generative ai",
]

FINETUNING_KEYWORDS = ["fine tuning", "fine-tuning", "lora", "qlora", "peft"]

EVALUATION_KEYWORDS = [
    "ndcg",
    "mrr",
    "map",
    "precision at",
    "recall at",
    "a/b testing",
    "ab testing",
    "offline evaluation",
    "online experiment",
    "ranking evaluation",
]

PRODUCTION_KEYWORDS = [
    "python",
    "fastapi",
    "flask",
    "api",
    "backend",
    "production",
    "deployment",
    "scalable",
    "distributed systems",
    "cloud",
    "docker",
    "kubernetes",
]

MLOPS_KEYWORDS = [
    "mlops",
    "kubeflow",
    "airflow",
    "wandb",
    "weights & biases",
    "model monitoring",
    "model serving",
    "ci/cd",
]

CONSULTING_SERVICE_COMPANIES = [
    "tcs",
    "infosys",
    "wipro",
    "accenture",
    "cognizant",
    "capgemini",
    "hcl",
    "tech mahindra",
    "mindtree",
    "lti",
    "ltfs",
    "mphasis",
]

PRODUCT_STARTUP_SIGNALS = [
    "product",
    "saas",
    "platform",
    "marketplace",
    "startup",
    "software",
    "internet",
    "ai-native",
    "transportation",
    "fintech",
    "e-commerce",
]

TARGET_SKILL_KEYWORDS = sorted(
    set(
        RETRIEVAL_KEYWORDS
        + RECOMMENDATION_KEYWORDS
        + VECTOR_KEYWORDS
        + LLM_NLP_KEYWORDS
        + FINETUNING_KEYWORDS
        + EVALUATION_KEYWORDS
        + PRODUCTION_KEYWORDS
        + MLOPS_KEYWORDS
        + [
            "machine learning",
            "deep learning",
            "py torch",
            "pytorch",
            "tensor flow",
            "tensorflow",
            "scikit learn",
        ]
    )
)

ADVANCED_TARGET_SKILL_KEYWORDS = sorted(
    set(
        RETRIEVAL_KEYWORDS
        + VECTOR_KEYWORDS
        + LLM_NLP_KEYWORDS
        + FINETUNING_KEYWORDS
        + EVALUATION_KEYWORDS
        + MLOPS_KEYWORDS
    )
)


def _normalize_keywords(keywords: list[str]) -> list[str]:
    """Normalize static keyword lists once for faster matching."""

    return [normalize_text(keyword) for keyword in keywords]


NORMALIZED_RETRIEVAL_KEYWORDS = _normalize_keywords(RETRIEVAL_KEYWORDS)
NORMALIZED_RECOMMENDATION_KEYWORDS = _normalize_keywords(RECOMMENDATION_KEYWORDS)
NORMALIZED_VECTOR_KEYWORDS = _normalize_keywords(VECTOR_KEYWORDS)
NORMALIZED_LLM_NLP_KEYWORDS = _normalize_keywords(LLM_NLP_KEYWORDS)
NORMALIZED_FINETUNING_KEYWORDS = _normalize_keywords(FINETUNING_KEYWORDS)
NORMALIZED_EVALUATION_KEYWORDS = _normalize_keywords(EVALUATION_KEYWORDS)
NORMALIZED_PRODUCTION_KEYWORDS = _normalize_keywords(PRODUCTION_KEYWORDS)
NORMALIZED_MLOPS_KEYWORDS = _normalize_keywords(MLOPS_KEYWORDS)
NORMALIZED_CONSULTING_SERVICE_COMPANIES = _normalize_keywords(CONSULTING_SERVICE_COMPANIES)
NORMALIZED_PRODUCT_STARTUP_SIGNALS = _normalize_keywords(PRODUCT_STARTUP_SIGNALS)
NORMALIZED_TARGET_SKILL_KEYWORDS = _normalize_keywords(TARGET_SKILL_KEYWORDS)
NORMALIZED_ADVANCED_TARGET_SKILL_KEYWORDS = _normalize_keywords(
    ADVANCED_TARGET_SKILL_KEYWORDS
)
NORMALIZED_RANKING_KEYWORDS = _normalize_keywords(
    ["ranking", "ranker", "reranking", "re-ranking", "learning to rank"]
)
NORMALIZED_EMBEDDING_KEYWORDS = _normalize_keywords(["embedding", "embeddings"])
NORMALIZED_PYTHON_KEYWORDS = _normalize_keywords(["python"])


def _as_dict(value: Any) -> dict[str, Any]:
    """Return value as a dict when possible."""

    return value if isinstance(value, dict) else {}


def _iter_dicts(value: Any) -> list[dict[str, Any]]:
    """Return dictionary items from a list-like field."""

    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _safe_number(value: Any) -> int | float | None:
    """Parse a simple numeric value without raising."""

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


def _safe_bool(value: Any) -> bool | None:
    """Parse common boolean values without guessing aggressively."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = normalize_text(value)
        if normalized in {"true", "yes", "y", "1"}:
            return True
        if normalized in {"false", "no", "n", "0"}:
            return False
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    return None


def _keyword_count(text: str, normalized_keywords: list[str]) -> int:
    """Count distinct keywords present in text."""

    if not text:
        return 0
    return sum(
        1 for keyword in normalized_keywords if keyword and keyword in text
    )


def count_keyword_hits(text: str, keywords: list[str]) -> int:
    """Count how many configured keywords appear in normalized candidate text."""

    return _keyword_count(normalize_text(text), _normalize_keywords(keywords))


def _has_any(normalized_text: str, normalized_keywords: list[str]) -> bool:
    """Return whether any keyword appears in text."""

    if not normalized_text:
        return False
    return any(keyword and keyword in normalized_text for keyword in normalized_keywords)


def _skill_matches(skill_name: str, normalized_keywords: list[str]) -> bool:
    """Check a normalized skill name against keyword evidence."""

    normalized = normalize_skill_name(skill_name)
    return any(keyword and keyword in normalized for keyword in normalized_keywords)


def _count_matching_skills(skill_names: list[str], normalized_keywords: list[str]) -> int:
    """Count skill names matching at least one target keyword."""

    return sum(
        1
        for skill_name in skill_names
        if _skill_matches(skill_name, normalized_keywords)
    )


def _is_expert_skill(skill: dict[str, Any]) -> bool:
    """Return whether a skill item claims high proficiency."""

    proficiency = normalize_text(skill.get("proficiency"))
    return proficiency in {"expert", "advanced", "senior", "master"}


def _duration_months(skill: dict[str, Any]) -> int | float | None:
    """Read a skill duration in months."""

    return _safe_number(skill.get("duration_months"))


def _current_role_duration_months(career_history: list[dict[str, Any]]) -> int | float | None:
    """Read duration for the current or first career item when present."""

    for item in career_history:
        if _safe_bool(item.get("is_current")) is True:
            return _safe_number(item.get("duration_months"))
    if career_history:
        return _safe_number(career_history[0].get("duration_months"))
    return None


def _company_text(profile: dict[str, Any], career_history: list[dict[str, Any]]) -> str:
    """Build a small text field for company and industry signals."""

    parts: list[str] = [
        str(profile.get("current_company", "")),
        str(profile.get("current_company_size", "")),
        str(profile.get("current_industry", "")),
    ]
    for item in career_history:
        parts.extend(
            [
                str(item.get("company", "")),
                str(item.get("industry", "")),
                str(item.get("company_size", "")),
            ]
        )
    return normalize_text(" ".join(parts))


def _get_signal(signals: dict[str, Any], key: str) -> Any:
    """Read a behavioral signal by key."""

    return signals.get(key)


def extract_features(candidate: dict[str, Any]) -> dict[str, Any]:
    """Extract flat ranking features from one Redrob candidate record."""

    candidate_with_processed = (
        candidate if isinstance(candidate.get("_processed"), dict) else preprocess_candidate(candidate)
    )
    processed = candidate_with_processed["_processed"]
    profile = _as_dict(candidate_with_processed.get("profile"))
    career_history = _iter_dicts(candidate_with_processed.get("career_history"))
    skills = _iter_dicts(candidate_with_processed.get("skills"))
    signals = _as_dict(candidate_with_processed.get("redrob_signals"))

    profile_text = processed.get("normalized_profile_text", "")
    career_text = processed.get("normalized_career_text", "")
    skill_text = processed.get("normalized_skill_text", "")
    education_text = processed.get("normalized_education_text", "")
    all_text = processed.get("normalized_all_text", "")
    normalized_skill_names = list(processed.get("normalized_skill_names", []))
    company_text = _company_text(profile, career_history)

    career_profile_text = career_text + " " + profile_text
    career_retrieval_keyword_count = _keyword_count(
        career_profile_text, NORMALIZED_RETRIEVAL_KEYWORDS
    )
    skill_retrieval_keyword_count = _keyword_count(
        skill_text, NORMALIZED_RETRIEVAL_KEYWORDS
    )
    career_vector_keyword_count = _keyword_count(
        career_profile_text, NORMALIZED_VECTOR_KEYWORDS
    )
    skill_vector_keyword_count = _keyword_count(skill_text, NORMALIZED_VECTOR_KEYWORDS)
    career_llm_nlp_keyword_count = _keyword_count(
        career_profile_text, NORMALIZED_LLM_NLP_KEYWORDS
    )
    skill_llm_nlp_keyword_count = _keyword_count(skill_text, NORMALIZED_LLM_NLP_KEYWORDS)
    career_evaluation_keyword_count = _keyword_count(
        career_profile_text, NORMALIZED_EVALUATION_KEYWORDS
    )
    skill_evaluation_keyword_count = _keyword_count(
        skill_text, NORMALIZED_EVALUATION_KEYWORDS
    )
    career_production_keyword_count = _keyword_count(
        career_profile_text, NORMALIZED_PRODUCTION_KEYWORDS
    )
    skill_production_keyword_count = _keyword_count(
        skill_text, NORMALIZED_PRODUCTION_KEYWORDS
    )

    retrieval_keyword_count = (
        career_retrieval_keyword_count + skill_retrieval_keyword_count
    )
    vector_keyword_count = career_vector_keyword_count + skill_vector_keyword_count
    llm_nlp_keyword_count = career_llm_nlp_keyword_count + skill_llm_nlp_keyword_count
    evaluation_keyword_count = (
        career_evaluation_keyword_count + skill_evaluation_keyword_count
    )
    production_keyword_count = (
        career_production_keyword_count + skill_production_keyword_count
    )

    expert_skill_count = sum(1 for skill in skills if _is_expert_skill(skill))
    zero_duration_expert_skill_count = sum(
        1
        for skill in skills
        if _is_expert_skill(skill) and _duration_months(skill) == 0
    )

    has_consulting_service_company = _has_any(
        company_text, NORMALIZED_CONSULTING_SERVICE_COMPANIES
    )
    career_company_texts: list[str] = []
    for item in career_history:
        career_company_texts.append(
            normalize_text(
                " ".join(
                    [
                        str(item.get("company", "")),
                        str(item.get("industry", "")),
                        str(item.get("company_size", "")),
                    ]
                )
            )
        )

    consulting_only_background = bool(career_company_texts) and all(
        _has_any(text, NORMALIZED_CONSULTING_SERVICE_COMPANIES)
        for text in career_company_texts
    )

    return {
        "candidate_id": candidate_with_processed.get(
            "candidate_id", candidate_with_processed.get("id")
        ),
        "years_of_experience": _safe_number(profile.get("years_of_experience")),
        "current_title": profile.get("current_title"),
        "location": profile.get("location"),
        "country": profile.get("country"),
        "normalized_skill_names": normalized_skill_names,
        "profile_text": profile_text,
        "career_text": career_text,
        "skill_text": skill_text,
        "education_text": education_text,
        "all_text": all_text,
        "has_retrieval_evidence": _has_any(
            career_profile_text, NORMALIZED_RETRIEVAL_KEYWORDS
        ),
        "has_ranking_evidence": _has_any(
            career_profile_text,
            NORMALIZED_RANKING_KEYWORDS,
        ),
        "has_recommendation_evidence": _has_any(
            career_profile_text, NORMALIZED_RECOMMENDATION_KEYWORDS
        ),
        "has_vector_db_evidence": _has_any(
            career_profile_text + " " + skill_text, NORMALIZED_VECTOR_KEYWORDS
        ),
        "has_embedding_evidence": _has_any(
            career_profile_text + " " + skill_text,
            NORMALIZED_EMBEDDING_KEYWORDS,
        ),
        "has_llm_nlp_evidence": _has_any(
            career_profile_text + " " + skill_text, NORMALIZED_LLM_NLP_KEYWORDS
        ),
        "has_finetuning_evidence": _has_any(all_text, NORMALIZED_FINETUNING_KEYWORDS),
        "has_python_evidence": _has_any(all_text, NORMALIZED_PYTHON_KEYWORDS),
        "has_backend_or_production_evidence": _has_any(
            all_text, NORMALIZED_PRODUCTION_KEYWORDS
        ),
        "has_evaluation_evidence": _has_any(all_text, NORMALIZED_EVALUATION_KEYWORDS),
        "has_mlops_evidence": _has_any(all_text, NORMALIZED_MLOPS_KEYWORDS),
        "career_retrieval_keyword_count": career_retrieval_keyword_count,
        "skill_retrieval_keyword_count": skill_retrieval_keyword_count,
        "career_vector_keyword_count": career_vector_keyword_count,
        "skill_vector_keyword_count": skill_vector_keyword_count,
        "career_llm_nlp_keyword_count": career_llm_nlp_keyword_count,
        "skill_llm_nlp_keyword_count": skill_llm_nlp_keyword_count,
        "career_evaluation_keyword_count": career_evaluation_keyword_count,
        "skill_evaluation_keyword_count": skill_evaluation_keyword_count,
        "career_production_keyword_count": career_production_keyword_count,
        "skill_production_keyword_count": skill_production_keyword_count,
        "retrieval_keyword_count": retrieval_keyword_count,
        "vector_keyword_count": vector_keyword_count,
        "llm_nlp_keyword_count": llm_nlp_keyword_count,
        "evaluation_keyword_count": evaluation_keyword_count,
        "production_keyword_count": production_keyword_count,
        "target_skill_count": _count_matching_skills(
            normalized_skill_names, NORMALIZED_TARGET_SKILL_KEYWORDS
        ),
        "advanced_target_skill_count": _count_matching_skills(
            normalized_skill_names, NORMALIZED_ADVANCED_TARGET_SKILL_KEYWORDS
        ),
        "expert_skill_count": expert_skill_count,
        "zero_duration_expert_skill_count": zero_duration_expert_skill_count,
        "current_company": profile.get("current_company"),
        "current_company_size": profile.get("current_company_size"),
        "current_industry": profile.get("current_industry"),
        "has_product_company_signal": _has_any(
            company_text, NORMALIZED_PRODUCT_STARTUP_SIGNALS
        ),
        "has_consulting_service_company": has_consulting_service_company,
        "consulting_only_background": consulting_only_background,
        "career_role_count": len(career_history),
        "current_role_duration_months": _current_role_duration_months(career_history),
        "profile_completeness_score": _safe_number(
            _get_signal(signals, "profile_completeness_score")
        ),
        "open_to_work_flag": _safe_bool(_get_signal(signals, "open_to_work_flag")),
        "recruiter_response_rate": _safe_number(
            _get_signal(signals, "recruiter_response_rate")
        ),
        "avg_response_time_hours": _safe_number(
            _get_signal(signals, "avg_response_time_hours")
        ),
        "notice_period_days": _safe_number(_get_signal(signals, "notice_period_days")),
        "preferred_work_mode": _get_signal(signals, "preferred_work_mode"),
        "willing_to_relocate": _safe_bool(_get_signal(signals, "willing_to_relocate")),
        "github_activity_score": _safe_number(_get_signal(signals, "github_activity_score")),
        "search_appearance_30d": _safe_number(_get_signal(signals, "search_appearance_30d")),
        "saved_by_recruiters_30d": _safe_number(
            _get_signal(signals, "saved_by_recruiters_30d")
        ),
        "interview_completion_rate": _safe_number(
            _get_signal(signals, "interview_completion_rate")
        ),
        "offer_acceptance_rate": _safe_number(_get_signal(signals, "offer_acceptance_rate")),
        "verified_email": _safe_bool(_get_signal(signals, "verified_email")),
        "verified_phone": _safe_bool(_get_signal(signals, "verified_phone")),
        "linkedin_connected": _safe_bool(_get_signal(signals, "linkedin_connected")),
        "parsed_signup_date": processed.get("parsed_signup_date"),
        "parsed_last_active_date": processed.get("parsed_last_active_date"),
        "salary_min": processed.get("salary_min"),
        "salary_max": processed.get("salary_max"),
    }


def extract_placeholder_features(text: str) -> dict[str, float]:
    """Build lightweight placeholder features for the starter ranker."""

    keywords = [
        "ai",
        "machine learning",
        "ml",
        "python",
        "llm",
        "deep learning",
        "nlp",
        "pytorch",
        "tensorflow",
        "rag",
        "production",
    ]
    return {
        "keyword_hits": float(count_keyword_hits(text, keywords)),
        "text_length": float(len(text)),
    }


if __name__ == "__main__":
    fake_candidate = {
        "candidate_id": "demo-1",
        "profile": {
            "headline": "Senior AI Engineer",
            "summary": "Built semantic search, RAG, and ranking systems in production.",
            "location": "Bengaluru",
            "country": "India",
            "years_of_experience": 8,
            "current_title": "Senior AI Engineer",
            "current_company": "Search Platform AI",
            "current_company_size": "51-200",
            "current_industry": "SaaS product",
        },
        "career_history": [
            {
                "title": "AI Engineer",
                "company": "Search Platform AI",
                "industry": "Software",
                "company_size": "51-200",
                "duration_months": 30,
                "is_current": True,
                "description": (
                    "Owned BM25 plus vector database retrieval, embeddings, "
                    "Python FastAPI services, and offline ranking evaluation."
                ),
            }
        ],
        "skills": [
            {"name": "Python", "proficiency": "expert", "duration_months": 72},
            {"name": "vector db", "proficiency": "advanced", "duration_months": 24},
            {"name": "sentence-transformers", "proficiency": "expert", "duration_months": 24},
        ],
        "education": [{"institution": "Example University", "degree": "MS"}],
        "redrob_signals": {
            "signup_date": "2024-01-15",
            "last_active_date": "2026-06-01",
            "expected_salary_range_inr_lpa": {"min": "45", "max": "70"},
            "profile_completeness_score": 92,
            "open_to_work_flag": True,
        },
    }

    features = extract_features(fake_candidate)
    assert features["has_retrieval_evidence"] is True
    assert features["has_vector_db_evidence"] is True
    assert features["has_python_evidence"] is True
    assert features["target_skill_count"] > 0
    print(
        {
            "candidate_id": features["candidate_id"],
            "has_retrieval_evidence": features["has_retrieval_evidence"],
            "has_vector_db_evidence": features["has_vector_db_evidence"],
            "career_retrieval_keyword_count": features["career_retrieval_keyword_count"],
            "skill_vector_keyword_count": features["skill_vector_keyword_count"],
            "target_skill_count": features["target_skill_count"],
        }
    )
