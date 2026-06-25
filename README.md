# Redrob India Runs — Candidate Ranking Engine

## Setup (Windows)

1. Install Python 3.10+ from python.org (check "Add Python to PATH" during install)
2. Open Command Prompt in this folder
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Run it

Copy your `candidates.jsonl` file (from the dataset zip) into the `data/` folder, then run:

```
cd src
python rank.py --candidates ..\data\candidates.jsonl --out ..\artifacts\submission.csv
```

This will take ~1-2 minutes on a normal laptop and produce `submission.csv` in the `artifacts` folder.

## Validate

Run the official validator (copy `validate_submission.py` from the dataset zip into this folder first):

```
python validate_submission.py artifacts\submission.csv
```
