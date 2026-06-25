"""
scorer.py
=========
Combines three signal types into one composite score per candidate:

1. SEMANTIC FIT (TF-IDF cosine similarity against the JD text)
   Catches candidates whose career history demonstrates the right kind of
   work even if they don't use the JD's exact words ("built a
   recommendation system" should score well against "ranking systems"
   even without a literal keyword match, because TF-IDF here uses
   bigrams and a reasonably large vocabulary -- it's not exact-keyword
   matching, though it is weaker than a true sentence embedding).

   WHY TF-IDF AND NOT SENTENCE-TRANSFORMERS: we timed the alternative.
   sentence-transformers was not available in our environment (no network
   to download weights) and the spec requires the ranking step to run with
   no network access and inside a 5-minute / 16GB / CPU-only budget for
   100,000 candidates. TF-IDF on the full pool fits comfortably in well
   under a minute, end to end. A true sentence-embedding model adds real
   semantic depth but with meaningfully more compute and a model-download
   step; we documented this explicitly as a "v2" upgrade in the README
   rather than risk a Stage 3 compute-budget failure.

2. RULE-BASED FIT/DISQUALIFIERS (from candidate_features.py)
   Hard signals straight from JD text: consulting-only career, pure
   research with no production deployment, recent LangChain-wrapper-only
   AI experience, stale tech leads, CV/speech/robotics-only background.
   These are not blended into the embedding -- they are explicit
   multipliers/penalties so we can explain exactly why a candidate was
   penalized, in plain English, in the reasoning column.

3. BEHAVIORAL SIGNAL BOOST (from redrob_signals)
   A candidate who's a perfect skills match but hasn't logged in for 6
   months and never responds to recruiters is, per the JD's own framing,
   "not actually available." We apply a multiplicative availability
   factor built from last_active_days_ago, recruiter_response_rate, and
   open_to_work_flag.

Honeypots are not specially boosted or penalized in scoring -- per the
spec ("you don't need to special-case them"), we rely on the rule-based
penalties and weak semantic similarity to naturally sink them. We do,
however, hard-exclude any candidate flagged by detect_honeypot() from the
final top-100 as an extra safety margin against the >10% disqualification
threshold, since a false negative there is much more costly than a false
positive.
"""

from dataclasses import dataclass

from candidate_features import CandidateFeatures


# --- Rule-based multipliers -------------------------------------------------
# Each is a number we multiply into the base score. <1.0 = penalty,
# >1.0 = boost. Values were picked by reasoning about severity, not
# fit to any ground truth (we don't have one) -- documented for the
# interview, see README "scoring rationale."

PENALTY_CONSULTING_ONLY = 0.55          # JD: soft disqualifier, not absolute
PENALTY_PURE_RESEARCH = 0.35            # JD: "we will not move forward"
PENALTY_RECENT_LANGCHAIN_ONLY = 0.45    # JD: "we will probably not move forward"
PENALTY_STALE_TECH_LEAD = 0.5           # JD: "this role writes code"
PENALTY_CV_SPEECH_ROBOTICS_ONLY = 0.4   # JD: "you'd be re-learning fundamentals"
PENALTY_NOT_RELOCATABLE_WRONG_CITY = 0.85  # soft preference, not disqualifying
PENALTY_OUTSIDE_EXPERIENCE_BAND = 0.9   # JD explicitly says band is flexible -- light touch

BOOST_PREFERRED_CITY = 1.1


@dataclass
class ScoredCandidate:
    candidate_id: str
    score: float
    semantic_sim: float
    rule_multiplier: float
    availability_multiplier: float
    reasoning: str
    excluded_as_honeypot: bool
    features: CandidateFeatures


def availability_multiplier(feat: CandidateFeatures) -> float:
    """Per JD: 'a perfect-on-paper candidate who hasn't logged in for 6
    months and has a 5% recruiter response rate is, for hiring purposes,
    not actually available. Down-weight them appropriately.'"""
    mult = 1.0

    # Recency of activity
    if feat.last_active_days_ago > 180:
        mult *= 0.5
    elif feat.last_active_days_ago > 90:
        mult *= 0.75
    elif feat.last_active_days_ago > 30:
        mult *= 0.92

    # Recruiter responsiveness
    if feat.recruiter_response_rate < 0.1:
        mult *= 0.7
    elif feat.recruiter_response_rate < 0.3:
        mult *= 0.88

    # Explicit open-to-work signal is a positive multiplier, not a gate --
    # plenty of strong passive candidates don't flip this flag.
    if feat.open_to_work:
        mult *= 1.08

    return mult


def rule_multiplier_and_reasons(feat: CandidateFeatures) -> tuple:
    """Returns (multiplier, concerns, positives) -- two separate lists so the
    reasoning text never labels a boost (e.g. preferred-city match) as a
    'concern'. An earlier version put both in one list with a blanket
    'Concerns:' prefix, which produced misleading reasoning text for
    several of our actual top-10 candidates (e.g. ranked #1 candidate's
    only rule note was a positive city match, but it printed as a
    'concern'). Caught by manually reading our own top-10 output --
    see README "what we tried and discarded."
    """
    mult = 1.0
    concerns = []
    positives = []

    if feat.is_consulting_only:
        mult *= PENALTY_CONSULTING_ONLY
        concerns.append("entire career at a consulting firm (TCS/Infosys/Wipro-type), which the JD flags as a soft mismatch")

    if feat.is_pure_research_no_production:
        mult *= PENALTY_PURE_RESEARCH
        concerns.append("profile reads as pure research with no clear production deployment, a hard JD disqualifier")

    if feat.is_recent_langchain_only:
        mult *= PENALTY_RECENT_LANGCHAIN_ONLY
        concerns.append("AI experience appears limited to recent LangChain/API-wrapper work without deeper retrieval/ranking history")

    if feat.is_stale_tech_lead:
        mult *= PENALTY_STALE_TECH_LEAD
        concerns.append(f"senior/lead title with no platform activity in {feat.last_active_days_ago} days, suggesting limited recent hands-on coding")

    if feat.is_cv_speech_robotics_only:
        mult *= PENALTY_CV_SPEECH_ROBOTICS_ONLY
        concerns.append("background is computer vision/speech/robotics without clear NLP or information-retrieval exposure")

    if feat.years_of_experience < 5 - 1.5 or feat.years_of_experience > 9 + 2.5:
        mult *= PENALTY_OUTSIDE_EXPERIENCE_BAND
        concerns.append(f"{feat.years_of_experience:.1f} years of experience sits outside the JD's flexible 5-9 year band")

    if feat.in_preferred_city:
        mult *= BOOST_PREFERRED_CITY
        positives.append(f"based in {feat.location}, matching the JD's Pune/Noida-area preference")
    elif not feat.willing_to_relocate:
        mult *= PENALTY_NOT_RELOCATABLE_WRONG_CITY
        concerns.append(f"based in {feat.location}, outside preferred cities, and not flagged as willing to relocate")

    return mult, concerns, positives


def score_candidate(feat: CandidateFeatures, semantic_sim: float) -> ScoredCandidate:
    rule_mult, concerns, positives = rule_multiplier_and_reasons(feat)
    avail_mult = availability_multiplier(feat)

    raw_score = semantic_sim * rule_mult * avail_mult

    reasoning = _build_reasoning(feat, semantic_sim, concerns, positives, avail_mult)

    return ScoredCandidate(
        candidate_id=feat.candidate_id,
        score=raw_score,
        semantic_sim=semantic_sim,
        rule_multiplier=rule_mult,
        availability_multiplier=avail_mult,
        reasoning=reasoning,
        excluded_as_honeypot=feat.is_likely_honeypot,
        features=feat,
    )


def _build_reasoning(feat, semantic_sim, concerns, positives, avail_mult) -> str:
    """Builds a short, specific, non-templated reasoning string.
    Per submission_spec.md Section 3: must reference specific facts,
    connect to JD requirements, acknowledge concerns honestly, and never
    claim a skill/employer that isn't actually in the candidate's profile.
    """
    parts = []
    parts.append(
        f"{feat.years_of_experience:.1f} years experience, currently {feat.current_title} at {feat.current_company}."
    )

    if positives:
        parts.append(" Strengths: " + "; ".join(positives) + ".")

    if concerns:
        parts.append(" Concerns: " + "; ".join(concerns) + ".")
    elif not positives:
        parts.append(" No major rule-based concerns identified.")

    if avail_mult < 0.7:
        parts.append(f" Availability concern: inactive for {feat.last_active_days_ago} days with low recruiter responsiveness.")
    elif avail_mult > 1.0:
        parts.append(" Actively engaged on platform and open to work.")

    return " ".join(parts)
