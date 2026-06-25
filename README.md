# Redrob AI — Intelligent Candidate Discovery & Ranking Engine

Built for **India Runs**, Track 1 (The Data & AI Challenge), Redrob AI /
Hack2Skill.

## What this does

Takes one job description (Senior AI Engineer, founding team, Redrob AI)
and ranks 100,000 candidate profiles against it, producing a top-100
shortlist with a short, specific, honest explanation for every rank.

The brief explicitly warns that naive keyword matching is the wrong
answer — the dataset contains candidates who match JD keywords perfectly
but fail the JD's real intent (pure researchers, consulting-only careers,
recent LangChain-wrapper-only AI work), plus a small number of
intentionally-impossible "honeypot" profiles. The system is built around
that constraint from the ground up. Full design rationale is in
[`ARCHITECTURE.md`](./ARCHITECTURE.md).

## Results (full 100,000-candidate run)

- **Runtime:** 60–120s end to end (budget: 300s), CPU only, no GPU, no
  network call, measured on two machines
- **Honeypots excluded:** ~45 of 100,000 (~0.045%), in the same order of
  magnitude as the spec's stated ~0.08% expected rate
- **Output:** top-100 ranked candidates, scores ranging ~0.23–0.55,
  passes the official `validate_submission.py` with zero errors

## How it works (short version)

Three signal types combine **multiplicatively** into one score per
candidate:

1. **Semantic fit** — TF-IDF cosine similarity between the candidate's
   full career text and the JD
2. **Rule-based fit** — explicit penalties/boosts for JD-specific
   disqualifiers (consulting-only, pure-research, LangChain-only,
   stale-tech-lead, CV/robotics-only, location)
3. **Availability** — behavioral signals (recency of activity, recruiter
   response rate, open-to-work flag)

Likely honeypots (profiles with internally impossible data, e.g. "expert"
proficiency with 0 months of use) are detected separately and excluded
outright before ranking. See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for
the full reasoning, including two real bugs we caught and fixed while
building this (an over-aggressive honeypot check, and a reasoning-text
labeling bug), documented honestly rather than smoothed over.

## Project structure

```
redrob-ranker/
├── src/
│   ├── jd_parser.py            # JD requirements, hand-encoded
│   ├── candidate_features.py   # feature extraction + honeypot detection
│   ├── scorer.py                # semantic + rule + behavioral scoring
│   └── rank.py                  # end-to-end orchestrator (CLI entry point)
├── data/                        # candidates.jsonl goes here (gitignored)
├── artifacts/                   # submission.csv is written here
├── ARCHITECTURE.md              # full system design + decision rationale
├── requirements.txt
└── README.md
```

## Setup

1. Install Python 3.10+ (during install on Windows, check **"Add Python
   to PATH"**)
2. From this folder:
   ```
   pip install -r requirements.txt
   ```

## Run it

Copy the organizer-provided `candidates.jsonl` into `data/`, then:

```
cd src
python rank.py --candidates ../data/candidates.jsonl --out ../artifacts/submission.csv
```

(On Windows, use backslashes: `..\data\candidates.jsonl` and
`..\artifacts\submission.csv`.)

This takes roughly 1–2 minutes on a normal laptop and writes
`submission.csv` to `artifacts/`.

## Validate

Copy the organizer-provided `validate_submission.py` into this folder,
then:

```
python validate_submission.py artifacts/submission.csv
```

Expected output: `Submission is valid.`

## Tech stack

Python, scikit-learn (`TfidfVectorizer`, `cosine_similarity`). No GPU, no
network calls, no LLM API calls at inference time — see
[`ARCHITECTURE.md` §5](./ARCHITECTURE.md#5-compute-budget-decision-tf-idf-over-sentence-transformers)
for why, and what a v2 with sentence embeddings would look like.