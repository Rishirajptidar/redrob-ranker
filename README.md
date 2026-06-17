# RecruiterRank-Hybrid

Redrob Hackathon solution for the **Intelligent Candidate Discovery & Ranking Challenge**.

The goal is to rank the top 100 candidates from `candidates.jsonl.gz` for the
**Senior AI Engineer -- Founding Team** role. The system is designed to run
offline, on CPU, and within the challenge memory/runtime constraints.

## Constraints

- CPU-only ranking
- No network calls during ranking
- No hosted LLM APIs
- No LangChain or LangGraph
- No GPU dependencies
- Streams candidates one by one instead of loading the full dataset
- Keeps memory bounded and below the 16 GB challenge limit
- Targets the 5-minute runtime limit on the organizer environment

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Reproduce Submission

```powershell
python src/rank.py --candidates data/candidates.jsonl.gz --out outputs/submission.csv
```

Validate the generated file:

```powershell
python validate_submission.py outputs/submission.csv
```

Expected output:

- `outputs/submission.csv`
- columns: `candidate_id`, `rank`, `score`, `reasoning`

## Architecture

1. **Data loading**

   `src/data_loader.py` streams `.jsonl` and `.jsonl.gz` records line by line.

2. **Preprocessing**

   `src/preprocessing.py` normalizes profile, career, skills, education, dates,
   salary range, location, and country fields using the nested Redrob schema.

3. **Feature extraction**

   `src/features.py` extracts technical, career, behavioral, company, location,
   and GitHub signals. Career/profile evidence is separated from skill-list
   evidence to reduce keyword-stuffing risk.

4. **Risk checks**

   `src/risk_checks.py` detects suspicious or low-confidence profiles, including:

   - salary min greater than max
   - signup date after last active date
   - skill stuffing without career evidence
   - irrelevant title with AI keywords
   - consulting-only background
   - weak activity or response signals
   - CV/speech/robotics mismatch without retrieval/NLP evidence

5. **Scoring**

   `src/scoring.py` combines technical fit, career evidence, experience fit,
   product/company fit, behavioral fit, location fit, GitHub fit, and risk
   penalties into a deterministic final score.

6. **Reasoning**

   `src/reasoning.py` creates short, grounded explanations for the final top
   candidates only.

7. **Ranking**

   `src/rank.py` streams all 100,000 candidates, scores each record, keeps only a
   bounded top candidate set with `heapq`, then writes the final top-100 CSV.

## Scoring Philosophy

- Career evidence is weighted more than skill-list keywords.
- Production retrieval, search, ranking, and recommendation experience is the
  strongest signal.
- Vector search and embedding evidence is strongly rewarded.
- Behavioral Redrob signals adjust hireability and reachability.
- Risk penalties reduce keyword-stuffers and honeypot-like profiles.

## Why This Is Not Simple Keyword Matching

RecruiterRank-Hybrid does not rank candidates only because they mention words like
"AI", "Python", or "LLM". The system separates weak skill-list matches from stronger
career-history evidence.

A candidate is ranked higher when the profile shows multiple matching signals together:

- relevant senior AI/ML role or career history
- production search, retrieval, ranking, recommendation, or NLP experience
- vector search, embeddings, or evaluation metric evidence
- Python/backend/product engineering background
- suitable experience range
- positive Redrob behavioral signals such as activity, response rate, open-to-work status, and recruiter interest
- low risk of keyword stuffing or inconsistent profile data

This makes the ranking closer to recruiter-style evaluation, where real work evidence,
seniority, availability, and trust signals matter more than raw keyword overlap.

## Runtime Notes

The pipeline processes all 100,000 candidates while keeping only a bounded heap
of high-scoring candidates in memory. On the local laptop, ranking completed
within the 5-minute challenge limit after optimization.

## Project Structure

```text
redrob-ranker/
├── data/
├── outputs/
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── risk_checks.py
│   ├── scoring.py
│   ├── reasoning.py
│   ├── rank.py
│   └── validation_utils.py
├── validate_submission.py
├── requirements.txt
├── submission_metadata.yaml
└── README.md
```

## AI Tools Disclosure

ChatGPT was used for architecture planning, code review, prompt assistance, and
documentation help. GitHub Copilot/Codex was used for implementation assistance.

The final ranking pipeline does not call any hosted LLM/API and works offline.
