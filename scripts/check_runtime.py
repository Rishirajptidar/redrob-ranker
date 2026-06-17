"""Run a quick local timing check for the ranking CLI."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time


def main() -> None:
    """Execute the ranker and print elapsed wall-clock time."""

    parser = argparse.ArgumentParser(description="Time the ranking pipeline.")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", default="outputs/submission.csv")
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()

    command = [
        sys.executable,
        "src/rank.py",
        "--candidates",
        args.candidates,
        "--out",
        args.out,
        "--top-k",
        str(args.top_k),
    ]

    start = time.perf_counter()
    subprocess.run(command, check=True)
    elapsed = time.perf_counter() - start
    print(f"elapsed_seconds={elapsed:.2f}")


if __name__ == "__main__":
    main()
