"""
app.py
======
Streamlit demo sandbox. Uses st.session_state to cache pipeline results
so that interacting with widgets (e.g. changing the candidate dropdown)
does NOT re-run the full pipeline or reset the file uploader.

Root cause of the original bug: Streamlit re-runs the entire script on
every widget interaction. Without session_state, the ranked results
were lost the moment the user touched any widget after the run completed,
sending the page back to the initial upload state. Fixed by storing
`top`, `timings`, and a `run_complete` flag in st.session_state after
the pipeline finishes, and reading from there on subsequent re-runs.
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
    "Upload `candidates.jsonl`, then click **Run ranking** to see the real "
    "pipeline execute end-to-end — semantic matching, rule-based JD checks, "
    "honeypot exclusion, and explainable scoring."
)

# --- Initialise session state keys we'll use ---
if "run_complete" not in st.session_state:
    st.session_state.run_complete = False
if "top" not in st.session_state:
    st.session_state.top = []
if "timings" not in st.session_state:
    st.session_state.timings = {}
if "df" not in st.session_state:
    st.session_state.df = None

col1, col2 = st.columns([2, 1])
with col1:
    uploaded = st.file_uploader("Upload candidates.jsonl", type=["jsonl"])
with col2:
    top_n = st.number_input(
        "How many top candidates to show",
        min_value=10, max_value=100, value=20, step=10
    )

run_clicked = st.button("▶️ Run ranking", type="primary")

# Reset state if user uploads a new file after a previous run
if uploaded is not None and st.session_state.run_complete:
    # Only reset if it's actually a new file (by size as a cheap proxy)
    if "last_file_size" not in st.session_state or \
       st.session_state.last_file_size != uploaded.size:
        st.session_state.run_complete = False
        st.session_state.top = []
        st.session_state.df = None

if run_clicked:
    if uploaded is None:
        st.warning("Please upload a candidates.jsonl file first.")
    else:
        tmp_path = str(Path(tempfile.gettempdir()) / "uploaded_candidates.jsonl")
        with open(tmp_path, "wb") as f:
            f.write(uploaded.getvalue())

        # Remember file size so we can detect re-uploads
        st.session_state.last_file_size = uploaded.size

        progress_box = st.empty()
        status_log = []

        def progress_callback(msg):
            status_log.append(msg)
            progress_box.info("\n\n".join(status_log))

        with st.spinner("Running pipeline..."):
            top, all_scored, timings = run_pipeline(
                tmp_path, top_n=int(top_n), progress_callback=progress_callback
            )

        progress_box.empty()

        # Store results in session state so they survive widget re-runs
        st.session_state.top = top
        st.session_state.timings = timings
        st.session_state.run_complete = True

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
        st.session_state.df = pd.DataFrame(rows)

# --- Display results if we have them (survives widget interactions) ---
if st.session_state.run_complete and st.session_state.df is not None:
    timings = st.session_state.timings
    top = st.session_state.top
    df = st.session_state.df

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Candidates processed", f"{timings['total_candidates']:,}")
    m2.metric("Honeypots excluded", timings["honeypots_excluded"])
    m3.metric("Runtime", f"{timings['total_s']:.1f}s")
    m4.metric("Budget", "300s",
              delta=f"{300 - timings['total_s']:.0f}s to spare",
              delta_color="normal")

    if timings["total_s"] > 300:
        st.error("⚠️ Exceeded the 5-minute compute budget!")
    else:
        st.success(f"✅ Completed within budget, {300 - timings['total_s']:.0f}s to spare.")

    st.markdown("---")
    st.subheader(f"Top {len(top)} ranked candidates")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Reasoning": st.column_config.TextColumn(width="large"),
            "Score": st.column_config.ProgressColumn(
                min_value=0,
                max_value=float(df["Score"].max()),
                format="%.3f"
            ),
        },
    )

    st.download_button(
        "⬇️ Download full results as CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="ranked_candidates.csv",
        mime="text/csv",
    )

    # --- This is the key fix: the dropdown now works without resetting the page ---
    with st.expander("🔍 Inspect one candidate's full reasoning"):
        selected_id = st.selectbox(
            "Candidate ID",
            df["Candidate ID"].tolist(),
            key="candidate_selector"   # explicit key prevents state collision
        )
        chosen = next((s for s in top if s.candidate_id == selected_id), None)
        if chosen:
            st.write(f"**Score breakdown for {selected_id}:**")
            st.write(f"- Semantic similarity: `{chosen.semantic_sim:.4f}`")
            st.write(f"- Rule-based multiplier: `{chosen.rule_multiplier:.4f}`")
            st.write(f"- Availability multiplier: `{chosen.availability_multiplier:.4f}`")
            st.write(f"- **Final score:** `{chosen.score:.4f}`")
            st.write(f"**Reasoning:** {chosen.reasoning}")

else:
    st.info("Upload a candidates.jsonl file and click **Run ranking** to begin.")