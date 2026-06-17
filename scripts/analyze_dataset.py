"""Lightweight dataset inspection script."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from data_loader import iter_candidates
from risk_checks import has_candidate_id


def main() -> None:
    """Print simple streaming dataset diagnostics."""

    parser = argparse.ArgumentParser(description="Analyze a Redrob JSONL dataset.")
    parser.add_argument("--candidates", required=True)
    args = parser.parse_args()

    count = 0
    missing_ids = 0
    for count, candidate in enumerate(iter_candidates(args.candidates), start=1):
        if not has_candidate_id(candidate):
            missing_ids += 1

    print(f"records={count}")
    print(f"missing_ids={missing_ids}")


if __name__ == "__main__":
    main()
