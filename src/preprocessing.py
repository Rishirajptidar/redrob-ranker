"""Deterministic preprocessing helpers for candidate records.

This module is intentionally standard-library only. Every function works on one
candidate at a time so the ranking pipeline can stream large JSONL files without
holding the dataset in memory.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any


_WHITESPACE_RE = re.compile(r"\s+")
_SEPARATOR_RE = re.compile(r"[\t\n\r|]+")

_SKILL_ALIASES = {
    "sentence-transformers": "sentence transformers",
    "vector db": "vector database",
    "vectordb": "vector database",
    "a/b testing": "ab testing",
    "llms": "llm",
    "huggingface": "hugging face",
    "opensearch": "open search",
    "elasticsearch": "elastic search",
    "sklearn": "scikit learn",
    "scikit-learn": "scikit learn",
    "pytorch": "py torch",
    "tensorflow": "tensor flow",
}


def normalize_text(text: str) -> str:
    """Lowercase text and normalize spacing without stripping useful punctuation."""

    if text is None:
        return ""
    value = str(text).lower()
    value = _SEPARATOR_RE.sub(" ", value)
    value = _WHITESPACE_RE.sub(" ", value)
    return value.strip()


def normalize_skill_name(skill: str) -> str:
    """Normalize a skill name and apply known aliases."""

    normalized = normalize_text(skill)
    return _SKILL_ALIASES.get(normalized, normalized)


def safe_parse_date(value: Any) -> date | None:
    """Parse a `YYYY-MM-DD` date, returning None for missing or invalid values."""

    if value is None:
        return None
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        return None

    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        return None


def _append_fields(parts: list[str], source: dict[str, Any], fields: list[str]) -> None:
    """Append present scalar field values from a dictionary."""

    for field in fields:
        value = source.get(field)
        if value is not None:
            parts.append(str(value))


def _iter_dicts(value: Any) -> list[dict[str, Any]]:
    """Return dictionary items from a list-like field."""

    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def build_profile_text(candidate: dict[str, Any]) -> str:
    """Build normalized text from top-level profile fields."""

    parts: list[str] = []
    profile = candidate.get("profile", {})
    if not isinstance(profile, dict):
        profile = {}

    _append_fields(
        parts,
        profile,
        [
            "headline",
            "summary",
            "current_title",
            "current_company",
            "current_company_size",
            "current_industry",
            "location",
            "country",
        ],
    )
    return normalize_text(" ".join(parts))


def build_career_text(candidate: dict[str, Any]) -> str:
    """Build normalized text from career history items."""

    parts: list[str] = []
    for item in _iter_dicts(candidate.get("career_history")):
        _append_fields(
            parts,
            item,
            ["title", "company", "industry", "company_size", "description"],
        )
    return normalize_text(" ".join(parts))


def build_skill_text(candidate: dict[str, Any]) -> str:
    """Build normalized text from skill items."""

    parts: list[str] = []
    for item in _iter_dicts(candidate.get("skills")):
        name = item.get("name")
        normalized_name = normalize_skill_name(name or item.get("normalized_name"))
        _append_fields(
            parts,
            item,
            ["name", "normalized_name", "proficiency", "duration_months", "endorsements"],
        )
        if normalized_name:
            parts.append(normalized_name)
    return normalize_text(" ".join(parts))


def build_education_text(candidate: dict[str, Any]) -> str:
    """Build normalized text from education items."""

    parts: list[str] = []
    for item in _iter_dicts(candidate.get("education")):
        _append_fields(
            parts,
            item,
            [
                "institution",
                "degree",
                "field_of_study",
                "grade",
                "tier",
                "start_year",
                "end_year",
            ],
        )
    return normalize_text(" ".join(parts))


def _safe_number(value: Any) -> int | float | None:
    """Return a numeric value, or None for missing and invalid values."""

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


def extract_salary_range(candidate: dict[str, Any]) -> tuple[int | float | None, int | float | None]:
    """Extract expected salary min and max in INR LPA without correcting values."""

    signals = candidate.get("redrob_signals")
    if not isinstance(signals, dict):
        return None, None

    salary_range = signals.get("expected_salary_range_inr_lpa")
    if not isinstance(salary_range, dict):
        return None, None

    return _safe_number(salary_range.get("min")), _safe_number(salary_range.get("max"))


def _normalized_skill_names(candidate: dict[str, Any]) -> list[str]:
    """Return normalized skill names, preserving candidate order."""

    names: list[str] = []
    for item in _iter_dicts(candidate.get("skills")):
        normalized = normalize_skill_name(item.get("name") or item.get("normalized_name"))
        if normalized:
            names.append(normalized)
    return names


def preprocess_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow candidate copy with derived normalized fields."""

    processed_candidate = dict(candidate)
    profile = candidate.get("profile", {})
    if not isinstance(profile, dict):
        profile = {}
    redrob_signals = candidate.get("redrob_signals", {})
    if not isinstance(redrob_signals, dict):
        redrob_signals = {}

    profile_text = build_profile_text(candidate)
    career_text = build_career_text(candidate)
    skill_text = build_skill_text(candidate)
    education_text = build_education_text(candidate)
    salary_min, salary_max = extract_salary_range(candidate)

    processed_candidate["_processed"] = {
        "normalized_profile_text": profile_text,
        "normalized_career_text": career_text,
        "normalized_skill_text": skill_text,
        "normalized_education_text": education_text,
        "normalized_all_text": normalize_text(
            " ".join([profile_text, career_text, skill_text, education_text])
        ),
        "normalized_skill_names": _normalized_skill_names(candidate),
        "parsed_signup_date": safe_parse_date(redrob_signals.get("signup_date")),
        "parsed_last_active_date": safe_parse_date(redrob_signals.get("last_active_date")),
        "salary_min": salary_min,
        "salary_max": salary_max,
        "current_title_normalized": normalize_text(profile.get("current_title")),
        "location_normalized": normalize_text(profile.get("location")),
        "country_normalized": normalize_text(profile.get("country")),
    }
    return processed_candidate


def candidate_to_text(candidate: dict[str, Any]) -> str:
    """Compatibility helper returning all normalized candidate text."""

    return preprocess_candidate(candidate)["_processed"]["normalized_all_text"]


if __name__ == "__main__":
    fake_candidate = {
        "profile": {
            "headline": "Senior AI Engineer",
            "summary": "Built LLMs, RAG systems, CI/CD, and A/B testing pipelines.",
            "location": "Bengaluru",
            "country": "India",
            "years_of_experience": 9,
            "current_title": "Lead ML Engineer",
            "current_company": "Example Labs",
            "current_company_size": "51-200",
            "current_industry": "SaaS",
        },
        "career_history": [
            {
                "title": "AI Engineer",
                "company": "Example Labs",
                "industry": "SaaS",
                "description": "Python, node.js, c++, and sentence-transformers.",
            }
        ],
        "skills": [
            {"name": "PyTorch", "proficiency": "expert", "duration_months": 48},
            {"name": "vector db", "endorsements": 12},
        ],
        "education": [
            {
                "institution": "Example University",
                "degree": "MS",
                "field_of_study": "Computer Science",
                "start_year": 2015,
                "end_year": 2017,
            }
        ],
        "redrob_signals": {
            "signup_date": "2024-01-15",
            "last_active_date": "bad-date",
            "expected_salary_range_inr_lpa": {"min": "45", "max": "70"},
        },
    }
    processed = preprocess_candidate(fake_candidate)
    assert "senior ai engineer" in processed["_processed"]["normalized_profile_text"]
    assert (
        "sentence-transformers" in processed["_processed"]["normalized_career_text"]
        or "sentence transformers" in processed["_processed"]["normalized_career_text"]
    )
    assert processed["_processed"]["parsed_signup_date"] is not None
    assert processed["_processed"]["salary_min"] == 45
    print(sorted(processed["_processed"].keys()))
