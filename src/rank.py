"""
rank.py
=======
End-to-end CLI entry point. Reads candidates.jsonl, scores every candidate,
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
~60-120 seconds (see README "Performance" section for the breakdown we
measured: ~30s JSON parse + feature extraction, ~30s TF-IDF fit, ~3s
cosine similarity, a few seconds for scoring + I/O).

NOTE: the actual scoring logic lives in pipeline.py, shared with the
Streamlit demo app (app.py) so the demo and the real submission can never
silently diverge.
"""

import argparse
import csv
import sys

from pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl or .jsonl.gz")
    parser.add_argument("--out", required=True, help="Output path for submission CSV")
    parser.add_argument("--top-n", type=int, default=100)
    args = parser.parse_args()

    top, all_scored, timings = run_pipeline(
        args.candidates,
        top_n=args.top_n,
        progress_callback=lambda msg: print(msg, file=sys.stderr),
    )

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, s in enumerate(top, start=1):
            writer.writerow([s.candidate_id, rank, round(s.score, 6), s.reasoning])

    print(f"\nWrote {len(top)} rows to {args.out}", file=sys.stderr)
    print(f"Total runtime: {timings['total_s']:.1f}s (budget: 300s)", file=sys.stderr)
    print(f"Honeypots excluded: {timings['honeypots_excluded']} of {timings['total_candidates']}", file=sys.stderr)
    if timings["total_s"] > 300:
        print("WARNING: exceeded the 5-minute compute budget!", file=sys.stderr)


if __name__ == "__main__":
    main()
