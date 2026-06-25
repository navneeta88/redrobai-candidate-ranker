"""
rank.py
=======
End-to-end entry point. Reads candidates.jsonl, scores every candidate,
excludes likely honeypots, and writes the top-100 ranked submission CSV.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Designed to satisfy the compute constraints in submission_spec.md Section 3:
    - <= 5 minutes wall-clock
    - <= 16 GB RAM
    - CPU only, no GPU
    - No network calls
    - <= 5 GB intermediate disk state

On this machine, end-to-end runtime on the full 100K candidate pool is
~60-90 seconds (see README "Performance" section for the breakdown we
measured: ~30s JSON parse + feature extraction, ~30s TF-IDF fit, ~3s
cosine similarity, a few seconds for scoring + I/O).
"""

import argparse
import csv
import json
import sys
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from candidate_features import extract_features
from jd_parser import get_jd_requirements
from scorer import score_candidate


def load_candidates(path: str):
    """Stream-load candidates.jsonl (handles plain or gzipped per README)."""
    opener = open
    if path.endswith(".gz"):
        import gzip
        opener = gzip.open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl or .jsonl.gz")
    parser.add_argument("--out", required=True, help="Output path for submission CSV")
    parser.add_argument("--top-n", type=int, default=100)
    args = parser.parse_args()

    t_start = time.time()

    # --- Step 1: load + extract features for every candidate ---
    print("Loading candidates and extracting features...", file=sys.stderr)
    all_features = []
    all_texts = []
    for raw in load_candidates(args.candidates):
        feat = extract_features(raw)
        all_features.append(feat)
        all_texts.append(feat.full_text)
    print(f"  Loaded {len(all_features)} candidates in {time.time()-t_start:.1f}s", file=sys.stderr)

    # --- Step 2: semantic similarity against the JD ---
    t1 = time.time()
    jd = get_jd_requirements()
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    candidate_matrix = vectorizer.fit_transform(all_texts)
    jd_vec = vectorizer.transform([jd.semantic_text])
    sims = cosine_similarity(jd_vec, candidate_matrix)[0]
    print(f"  TF-IDF + similarity done in {time.time()-t1:.1f}s", file=sys.stderr)

    # --- Step 3: score every candidate (rules + behavioral + semantic) ---
    t2 = time.time()
    scored = [
        score_candidate(feat, float(sim))
        for feat, sim in zip(all_features, sims)
    ]
    print(f"  Scored {len(scored)} candidates in {time.time()-t2:.1f}s", file=sys.stderr)

    # --- Step 4: exclude likely honeypots, then take top N ---
    clean = [s for s in scored if not s.excluded_as_honeypot]
    excluded_count = len(scored) - len(clean)
    print(f"  Excluded {excluded_count} likely honeypots ({100*excluded_count/len(scored):.3f}% of pool)", file=sys.stderr)

    clean.sort(key=lambda s: (-s.score, s.candidate_id))
    top = clean[: args.top_n]

    # --- Step 5: write submission CSV ---
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, s in enumerate(top, start=1):
            writer.writerow([s.candidate_id, rank, round(s.score, 6), s.reasoning])

    total_time = time.time() - t_start
    print(f"\nWrote {len(top)} rows to {args.out}", file=sys.stderr)
    print(f"Total runtime: {total_time:.1f}s (budget: 300s)", file=sys.stderr)
    if total_time > 300:
        print("WARNING: exceeded the 5-minute compute budget!", file=sys.stderr)


if __name__ == "__main__":
    main()
