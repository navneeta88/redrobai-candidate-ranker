"""
app.py
======
Streamlit demo sandbox. Lets a judge/reviewer upload (or use the bundled)
candidates.jsonl, click Run, and watch the real pipeline (pipeline.py --
the exact same code rank.py uses for the actual submission) execute live,
then browse the ranked, explainable results in a table instead of a CSV.

This is a demo wrapper, not a separate implementation -- it imports and
calls run_pipeline() directly. There is no scoring logic in this file.

Run locally:
    streamlit run app.py
"""

import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, ".")
from pipeline import run_pipeline
from jd_parser import JD_SEMANTIC_TEXT

st.set_page_config(page_title="Redrob Candidate Ranker", layout="wide")

st.title("🎯 Intelligent Candidate Discovery & Ranking Engine")
st.caption("India Runs — Track 1, Data & AI Challenge — Redrob AI")

with st.expander("📋 The job description this engine ranks against", expanded=False):
    st.text(JD_SEMANTIC_TEXT.strip())

st.markdown(
    "Upload `candidates.jsonl`, or use the small bundled sample, then click "
    "**Run ranking** to see the real pipeline execute end-to-end -- semantic "
    "matching, rule-based JD checks, honeypot exclusion, and explainable "
    "scoring -- exactly as it runs for the actual submission."
)

col1, col2 = st.columns([2, 1])
with col1:
    uploaded = st.file_uploader("Upload candidates.jsonl", type=["jsonl"])
with col2:
    top_n = st.number_input("How many top candidates to show", min_value=10, max_value=100, value=20, step=10)

run_clicked = st.button("▶️ Run ranking", type="primary")

if run_clicked:
    if uploaded is None:
        st.warning("Please upload a candidates.jsonl file first.")
    else:
        # Streamlit's uploader gives us an in-memory file; the pipeline
        # expects a path, so we write it to a temp file once. Using
        # tempfile.gettempdir() rather than a hardcoded /tmp path because
        # Windows has no /tmp -- this bug surfaced on first real Windows
        # test (FileNotFoundError: '/tmp/uploaded_candidates.jsonl').
        tmp_path = str(Path(tempfile.gettempdir()) / "uploaded_candidates.jsonl")
        with open(tmp_path, "wb") as f:
            f.write(uploaded.getvalue())

        progress_box = st.empty()
        status_log = []

        def progress_callback(msg):
            status_log.append(msg)
            progress_box.info("\n\n".join(status_log))

        t0 = time.time()
        with st.spinner("Running pipeline..."):
            top, all_scored, timings = run_pipeline(
                tmp_path, top_n=int(top_n), progress_callback=progress_callback
            )
        elapsed = time.time() - t0

        progress_box.empty()

        # --- Summary metrics ---
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Candidates processed", f"{timings['total_candidates']:,}")
        m2.metric("Honeypots excluded", timings["honeypots_excluded"])
        m3.metric("Runtime", f"{timings['total_s']:.1f}s")
        m4.metric("Budget", "300s", delta=f"{300 - timings['total_s']:.0f}s to spare", delta_color="normal")

        if timings["total_s"] > 300:
            st.error("⚠️ Exceeded the 5-minute compute budget!")
        else:
            st.success(f"✅ Completed within budget, with {300 - timings['total_s']:.0f}s to spare.")

        st.markdown("---")
        st.subheader(f"Top {len(top)} ranked candidates")

        rows = []
        for rank, s in enumerate(top, start=1):
            rows.append({
                "Rank": rank,
                "Candidate ID": s.candidate_id,
                "Score": round(s.score, 4),
                "Title": s.features.current_title,
                "Company": s.features.current_company,
                "Years Exp.": s.features.years_of_experience,
                "Location": s.features.location,
                "Reasoning": s.reasoning,
            })
        df = pd.DataFrame(rows)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Reasoning": st.column_config.TextColumn(width="large"),
                "Score": st.column_config.ProgressColumn(min_value=0, max_value=float(df["Score"].max()), format="%.3f"),
            },
        )

        st.download_button(
            "⬇️ Download full results as CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="ranked_candidates.csv",
            mime="text/csv",
        )

        with st.expander("🔍 Inspect one candidate's full reasoning"):
            selected_id = st.selectbox("Candidate ID", df["Candidate ID"].tolist())
            chosen = next(s for s in top if s.candidate_id == selected_id)
            st.write(f"**Score breakdown for {selected_id}:**")
            st.write(f"- Semantic similarity: `{chosen.semantic_sim:.4f}`")
            st.write(f"- Rule-based multiplier: `{chosen.rule_multiplier:.4f}`")
            st.write(f"- Availability multiplier: `{chosen.availability_multiplier:.4f}`")
            st.write(f"- **Final score:** `{chosen.score:.4f}`")
            st.write(f"**Reasoning:** {chosen.reasoning}")
else:
    st.info("Upload a candidates.jsonl file and click **Run ranking** to begin.")