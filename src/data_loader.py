"""Streaming candidate loading utilities.

The ranking pipeline must handle large JSONL files without loading the full
dataset into memory. This module exposes a small generator API for plain and
gzipped JSONL candidate files.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def iter_candidates(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield candidate dictionaries from a `.jsonl` or `.jsonl.gz` file.

    Empty lines are ignored. Each non-empty line must contain one JSON object.

    Args:
        path: Path to a candidate file ending in `.jsonl` or `.jsonl.gz`.

    Raises:
        FileNotFoundError: If the path does not exist.
        IsADirectoryError: If the path points to a directory.
        ValueError: If the extension is unsupported or a line is invalid.
    """

    candidate_path = Path(path)
    if not candidate_path.exists():
        raise FileNotFoundError(f"Candidate file not found: {candidate_path}")
    if candidate_path.is_dir():
        raise IsADirectoryError(f"Candidate path is a directory: {candidate_path}")

    suffix = "".join(candidate_path.suffixes[-2:])
    if candidate_path.suffix == ".jsonl":
        opener = open
    elif suffix == ".jsonl.gz":
        opener = gzip.open
    else:
        raise ValueError(
            "Candidate file must end with '.jsonl' or '.jsonl.gz': "
            f"{candidate_path}"
        )

    with opener(candidate_path, mode="rt", encoding="utf-8-sig") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {candidate_path}: "
                    f"{exc.msg}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Line {line_number} of {candidate_path} is not a JSON object."
                )

            yield record
