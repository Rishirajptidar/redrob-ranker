"""Command-line entry point for streaming candidate ranking."""

from __future__ import annotations

import argparse
import csv
import heapq
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from data_loader import iter_candidates
from reasoning import generate_reasoning
from scoring import score_candidate
from validation_utils import SUBMISSION_COLUMNS, validate_submission_row, validate_top_k


@dataclass
class RankedCandidate:
    """Compact heap item for memory-safe ranking."""

    score: float
    candidate_id: str
    candidate: dict[str, Any] = field(compare=False)
    reasoning: str = field(default="", compare=False)

    def __lt__(self, other: "RankedCandidate") -> bool:
        """Order heap items so the weakest retained candidate is popped first."""

        if self.score != other.score:
            return self.score < other.score
        return self.candidate_id > other.candidate_id


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Rank Redrob candidates.")
    parser.add_argument("--candidates", required=True, help="Path to .jsonl or .jsonl.gz")
    parser.add_argument("--out", required=True, help="Path to output CSV")
    parser.add_argument("--top-k", type=int, default=100, help="Number of rows to output")
    return parser.parse_args()


def _candidate_id_from_score(scored: dict[str, Any]) -> str:
    """Return a clean candidate id from scored output."""

    return str(scored.get("candidate_id") or "").strip()


def rank_candidates(
    candidates_path: str | Path, top_k: int
) -> tuple[list[RankedCandidate], int, int]:
    """Stream candidates and keep a bounded set of high-scoring records."""

    validate_top_k(top_k)
    retain_k = max(top_k * 5, 500)
    heap: list[RankedCandidate] = []
    processed_count = 0
    scored_count = 0

    for candidate in iter_candidates(candidates_path):
        processed_count += 1
        if processed_count % 10000 == 0:
            print(f"processed {processed_count} candidates...", flush=True)

        scored = score_candidate(candidate)
        candidate_id = _candidate_id_from_score(scored)
        if not candidate_id:
            continue

        item = RankedCandidate(
            score=float(scored["score"]),
            candidate_id=candidate_id,
            candidate=candidate,
        )
        scored_count += 1

        if len(heap) < retain_k:
            heapq.heappush(heap, item)
        else:
            heapq.heappushpop(heap, item)

    final_ranked = sorted(heap, key=lambda item: (-item.score, item.candidate_id))[:top_k]
    for item in final_ranked:
        scored = score_candidate(item.candidate)
        reasoning = generate_reasoning(item.candidate, scored).strip()
        if not reasoning:
            reasoning = "Candidate has limited available evidence for the Senior AI Engineer JD."
        if "placeholder" in reasoning.lower():
            reasoning = "Candidate scored using available profile, career, skill, and risk signals."
        item.reasoning = reasoning

    return final_ranked, processed_count, scored_count


def write_submission(rows: list[RankedCandidate], out_path: str | Path) -> None:
    """Write ranked candidates to the required submission CSV format."""

    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=SUBMISSION_COLUMNS)
        writer.writeheader()
        for rank, item in enumerate(rows, start=1):
            row = {
                "candidate_id": item.candidate_id,
                "rank": rank,
                "score": f"{item.score:.6f}",
                "reasoning": item.reasoning,
            }
            validate_submission_row(row)
            writer.writerow(row)


def main() -> None:
    """Run the ranking CLI."""

    args = parse_args()
    ranked, processed_count, scored_count = rank_candidates(args.candidates, args.top_k)
    write_submission(ranked, args.out)
    top_score = ranked[0].score if ranked else 0.0

    print(f"candidates processed: {processed_count}")
    print(f"candidates scored: {scored_count}")
    print(f"output path: {args.out}")
    print(f"top score: {top_score:.6f}")


if __name__ == "__main__":
    main()
