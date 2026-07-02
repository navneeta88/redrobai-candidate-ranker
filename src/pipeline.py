"""
pipeline.py
===========
Core pipeline logic, extracted so both rank.py (CLI, used for the actual
submission.csv) and app.py (Streamlit demo sandbox) call the exact same
code path. This matters: a demo that runs different logic than the real
submission would be misleading to show judges, even unintentionally.
"""

import json
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from candidate_features import extract_features
from jd_parser import get_jd_requirements
from scorer import score_candidate


def load_candidates(path: str):
    """Stream-load candidates.jsonl (handles plain or gzipped)."""
    opener = open
    if path.endswith(".gz"):
        import gzip
        opener = gzip.open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def run_pipeline(candidates_path: str, top_n: int = 100, progress_callback=None):
    """Runs the full pipeline and returns (top_scored_list, all_scored_list, timing_dict).

    progress_callback, if given, is called with a short status string after
    each major step -- used by the Streamlit app to update its UI live.
    It is never required by rank.py, which just ignores it (default None).
    """
    timings = {}
    t_start = time.time()

    def _report(msg):
        if progress_callback:
            progress_callback(msg)

    # Step 1: load + extract features
    _report("Loading candidates and extracting features...")
    all_features = []
    all_texts = []
    for raw in load_candidates(candidates_path):
        feat = extract_features(raw)
        all_features.append(feat)
        all_texts.append(feat.full_text)
    timings["load_and_extract_s"] = time.time() - t_start

    # Step 2: semantic similarity
    t1 = time.time()
    _report(f"Loaded {len(all_features)} candidates. Computing semantic similarity...")
    jd = get_jd_requirements()
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    candidate_matrix = vectorizer.fit_transform(all_texts)
    jd_vec = vectorizer.transform([jd.semantic_text])
    sims = cosine_similarity(jd_vec, candidate_matrix)[0]
    timings["tfidf_similarity_s"] = time.time() - t1

    # Step 3: score every candidate
    t2 = time.time()
    _report("Scoring candidates (rules + behavioral signals)...")
    scored = [
        score_candidate(feat, float(sim))
        for feat, sim in zip(all_features, sims)
    ]
    timings["scoring_s"] = time.time() - t2

    # Step 4: exclude honeypots, rank
    _report("Excluding likely honeypots and ranking...")
    clean = [s for s in scored if not s.excluded_as_honeypot]
    excluded_count = len(scored) - len(clean)
    clean.sort(key=lambda s: (-s.score, s.candidate_id))
    top = clean[:top_n]

    timings["total_s"] = time.time() - t_start
    timings["total_candidates"] = len(scored)
    timings["honeypots_excluded"] = excluded_count

    _report(f"Done. {len(top)} ranked, {excluded_count} honeypots excluded.")

    return top, scored, timings
