"""Streamlit sandbox demo for RecruiterRank-Hybrid."""

from __future__ import annotations

import csv
import gzip
import io
import json
import sys
from pathlib import Path
from typing import Any

import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.reasoning import generate_reasoning
from src.scoring import score_candidate


SUBMISSION_COLUMNS = ["candidate_id", "rank", "score", "reasoning"]
MAX_UPLOAD_CANDIDATES = 100


def _candidate_records_from_json(data: Any) -> list[dict[str, Any]]:
    """Validate JSON candidate data as one object or a list of objects."""

    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        records: list[dict[str, Any]] = []
        for index, item in enumerate(data, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"JSON array item {index} is not a candidate object.")
            records.append(item)
        return records
    raise ValueError("JSON upload must contain a candidate object or a list of objects.")


def _read_json_upload(uploaded_file: Any) -> list[dict[str, Any]]:
    """Read `.json` uploads containing one candidate or a candidate list."""

    uploaded_file.seek(0)
    raw_text = uploaded_file.getvalue().decode("utf-8-sig")
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {exc.msg}") from exc
    return _candidate_records_from_json(data)


def _read_jsonl_stream(stream: io.TextIOBase) -> list[dict[str, Any]]:
    """Read all candidate objects from a JSONL text stream."""

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(stream, start=1):
        line = line.strip()
        if not line:
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}: {exc.msg}") from exc

        if not isinstance(record, dict):
            raise ValueError(f"Line {line_number} is not a JSON object.")

        records.append(record)
    return records


def _read_jsonl_upload(uploaded_file: Any) -> list[dict[str, Any]]:
    """Read `.jsonl` or `.jsonl.gz` uploads line by line."""

    uploaded_file.seek(0)
    if uploaded_file.name.endswith(".jsonl.gz"):
        with gzip.GzipFile(fileobj=uploaded_file, mode="rb") as gzip_stream:
            with io.TextIOWrapper(gzip_stream, encoding="utf-8-sig") as stream:
                return _read_jsonl_stream(stream)

    with io.TextIOWrapper(uploaded_file, encoding="utf-8-sig") as stream:
        return _read_jsonl_stream(stream)


def load_candidates_from_upload(uploaded_file: Any) -> tuple[list[dict[str, Any]], bool]:
    """Load candidate records from `.json`, `.jsonl`, or `.jsonl.gz` uploads."""

    filename = uploaded_file.name.lower()
    if filename.endswith(".json"):
        candidates = _read_json_upload(uploaded_file)
    elif filename.endswith(".jsonl") or filename.endswith(".jsonl.gz"):
        candidates = _read_jsonl_upload(uploaded_file)
    else:
        raise ValueError("Please upload a .json, .jsonl, or .jsonl.gz file.")

    truncated = len(candidates) > MAX_UPLOAD_CANDIDATES
    return candidates[:MAX_UPLOAD_CANDIDATES], truncated


def _load_uploaded_candidates(uploaded_file: Any) -> tuple[list[dict[str, Any]], bool]:
    """Backward-compatible wrapper for upload loading."""

    return load_candidates_from_upload(uploaded_file)


def _rank_uploaded_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Score, rank, and explain uploaded candidates."""

    ranked_items: list[dict[str, Any]] = []
    for candidate in candidates:
        scored = score_candidate(candidate)
        candidate_id = str(scored.get("candidate_id") or "").strip()
        if not candidate_id:
            continue

        reasoning = generate_reasoning(candidate, scored).strip()
        if not reasoning:
            reasoning = "Candidate has limited available evidence for the Senior AI Engineer JD."

        ranked_items.append(
            {
                "candidate_id": candidate_id,
                "score": float(scored["score"]),
                "reasoning": reasoning,
            }
        )

    ranked_items.sort(key=lambda item: (-item["score"], item["candidate_id"]))

    rows: list[dict[str, str]] = []
    for rank, item in enumerate(ranked_items, start=1):
        rows.append(
            {
                "candidate_id": item["candidate_id"],
                "rank": str(rank),
                "score": f"{item['score']:.6f}",
                "reasoning": item["reasoning"],
            }
        )
    return rows


def _rows_to_csv(rows: list[dict[str, str]]) -> str:
    """Serialize ranked rows to CSV text."""

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=SUBMISSION_COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _show_overview() -> None:
    """Render project overview when no sample is uploaded."""

    st.subheader("RecruiterRank-Hybrid")
    st.write(
        "Offline candidate ranking system for the Redrob Senior AI Engineer "
        "Founding Team role. The full pipeline streams candidate records, "
        "extracts evidence-based features, applies risk checks, and writes a "
        "ranked submission CSV."
    )

    st.markdown(
        """
**Architecture flow**

`JSON/JSONL/GZ sample -> preprocessing -> feature extraction -> risk checks -> scoring -> ranking -> grounded reasoning -> CSV`

**Methodology summary**

RecruiterRank-Hybrid scores technical fit, career evidence, experience fit,
product/company fit, behavioral availability, location fit, and GitHub activity.
Career/profile evidence is weighted more than skill-list keywords to reduce
keyword-stuffing and honeypot-like profile risk.

This sandbox is for small-sample reproducibility and supports `.json`, `.jsonl`,
and `.jsonl.gz` uploads. The full 100k-candidate run should be reproduced locally.

**Runtime and constraint compliance**

- CPU-only
- No network calls during ranking
- No hosted LLM APIs
- No GPU
- Full local pipeline supports all 100,000 candidates with streaming and bounded heap retention

**Full reproduction**

```powershell
python src/rank.py --candidates data/candidates.jsonl.gz --out outputs/submission.csv
```
"""
    )


st.set_page_config(page_title="RecruiterRank-Hybrid", layout="wide")
st.title("RecruiterRank-Hybrid")
st.caption("Redrob candidate-ranking sandbox demo")

uploaded_file = st.file_uploader(
    "Upload a small candidate sample file",
    type=["json", "jsonl", "gz"],
    help="Supports .json, .jsonl, and .jsonl.gz. Recommended maximum: 100 candidate records.",
)

if uploaded_file is None:
    _show_overview()
else:
    if not (
        uploaded_file.name.endswith(".json")
        or uploaded_file.name.endswith(".jsonl")
        or uploaded_file.name.endswith(".jsonl.gz")
    ):
        st.error("Please upload a .json, .jsonl, or .jsonl.gz file.")
    else:
        try:
            candidates, truncated = _load_uploaded_candidates(uploaded_file)
        except OSError as exc:
            st.error(f"Could not read uploaded file: {exc}")
        except ValueError as exc:
            st.error(str(exc))
        else:
            if not candidates:
                st.warning("The uploaded file did not contain any candidate records.")
            else:
                if truncated:
                    st.warning(
                        "Uploaded file contains more than 100 candidates. Processing "
                        "first 100 for sandbox demo."
                    )

                rows = _rank_uploaded_candidates(candidates)
                if not rows:
                    st.warning("No ranked rows were produced. Check candidate_id fields.")
                else:
                    csv_text = _rows_to_csv(rows)
                    st.success(f"Ranked {len(rows)} candidates from the uploaded sample.")
                    st.dataframe(rows, width="stretch")
                    st.download_button(
                        "Download ranked CSV",
                        data=csv_text,
                        file_name="ranked_sample.csv",
                        mime="text/csv",
                    )
