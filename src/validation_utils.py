"""Output validation utilities."""

from __future__ import annotations


SUBMISSION_COLUMNS = ["candidate_id", "rank", "score", "reasoning"]


def validate_top_k(top_k: int) -> None:
    """Validate requested number of ranked candidates."""

    if top_k <= 0:
        raise ValueError("--top-k must be a positive integer.")


def validate_submission_row(row: dict[str, object]) -> None:
    """Validate that a row contains exactly the required output columns."""

    if list(row.keys()) != SUBMISSION_COLUMNS:
        raise ValueError(f"Submission row must have columns: {SUBMISSION_COLUMNS}")
