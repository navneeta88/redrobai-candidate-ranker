"""
candidate_features.py
======================
Turns a raw candidate JSON record into a flat, structured feature object.

Also implements honeypot detection: the JD/spec tells us the dataset
contains ~80 honeypot candidates with subtly impossible profiles (e.g.
8 years of experience at a company founded 3 years ago; "expert"
proficiency with 0 months used). We don't get a ground-truth honeypot
flag -- we have to find these via internal-consistency checks on each
profile. Catching them matters a lot: ranking honeypots in the top 100 at
>10% gets the whole submission disqualified at Stage 3.

We are deliberately NOT trying to build a perfect honeypot classifier.
We're building a small number of high-confidence internal-consistency
checks. Anything we flag should be flagged for a reason we can explain
in the interview -- "candidate claims expert proficiency in a skill they
list as having used for 0 months" is defensible. A vague ML anomaly score
is not.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

from jd_parser import (
    CONSULTING_FIRMS, RETRIEVAL_KEYWORDS, VECTOR_DB_KEYWORDS,
    EVAL_FRAMEWORK_KEYWORDS, LANGCHAIN_WRAPPER_KEYWORDS, NLP_IR_KEYWORDS,
    CV_SPEECH_ROBOTICS_KEYWORDS, TIER1_CITIES_PREFERRED,
)

TODAY = date(2026, 6, 19)  # pinned "as of" date for the challenge, see README


def _parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        return None


def _full_career_text(candidate: dict) -> str:
    """Concatenate everything text-bearing about a candidate's career."""
    parts = [
        candidate["profile"].get("headline", ""),
        candidate["profile"].get("summary", ""),
        candidate["profile"].get("current_title", ""),
        candidate["profile"].get("current_industry", ""),
    ]
    for job in candidate.get("career_history", []):
        parts.append(job.get("title", ""))
        parts.append(job.get("description", ""))
        parts.append(job.get("industry", ""))
    for skill in candidate.get("skills", []):
        parts.append(skill.get("name", ""))
    return " | ".join(p for p in parts if p).lower()


@dataclass
class CandidateFeatures:
    candidate_id: str
    years_of_experience: float
    current_title: str
    current_company: str
    location: str
    country: str

    # Hard-disqualifier / rule-based signals
    is_consulting_only: bool
    is_pure_research_no_production: bool
    is_recent_langchain_only: bool
    is_stale_tech_lead: bool  # senior, no prod code in 18mo
    is_cv_speech_robotics_only: bool
    in_preferred_city: bool
    willing_to_relocate: bool

    # Honeypot signals
    honeypot_flags: list = field(default_factory=list)
    is_likely_honeypot: bool = False

    # Behavioral signals (carried through for the scorer)
    last_active_days_ago: int = 9999
    recruiter_response_rate: float = 0.0
    interview_completion_rate: float = 0.0
    open_to_work: bool = False
    notice_period_days: int = 999
    github_activity_score: float = -1.0
    profile_completeness: float = 0.0

    # Text blob for embedding
    full_text: str = ""


def detect_honeypot(candidate: dict) -> list:
    """Return a list of human-readable reasons this candidate looks impossible.
    Empty list = no red flags found."""
    flags = []
    profile = candidate["profile"]
    yoe = profile.get("years_of_experience", 0)

    # Check 1: tenure at a company longer than years_of_experience allows,
    # OR overlapping/impossible date math within career_history.
    total_months_claimed = sum(j.get("duration_months", 0) for j in candidate.get("career_history", []))
    if total_months_claimed > (yoe * 12) + 12:  # +12mo slack for rounding/gaps
        flags.append(
            f"career_history totals {total_months_claimed} months but "
            f"years_of_experience is only {yoe} ({yoe*12:.0f} months)"
        )

    # Check 2: "expert" proficiency in a skill used for 0 (or near-0) months.
    for skill in candidate.get("skills", []):
        if skill.get("proficiency") == "expert" and skill.get("duration_months", 0) <= 1:
            flags.append(
                f"claims 'expert' in {skill.get('name')} with "
                f"{skill.get('duration_months', 0)} months of use"
            )

    # Check 3: current job duration_months alone exceeds total claimed YOE.
    for job in candidate.get("career_history", []):
        if job.get("duration_months", 0) > (yoe * 12) + 6:
            flags.append(
                f"single role at {job.get('company')} claims "
                f"{job.get('duration_months')} months, exceeding total YOE of {yoe}"
            )

    # NOTE: we deliberately do NOT flag "education ends after career starts."
    # Part-time / executive / distance degrees completed while already
    # working are extremely common in this dataset (and in India generally)
    # and are not evidence of an impossible profile. An earlier version of
    # this check produced an 18% false-positive rate on a 2,000-candidate
    # sample -- two orders of magnitude above the ~0.08% honeypot rate the
    # spec describes -- so it was removed. See README "what we tried and
    # discarded" section.

    return flags


def extract_features(candidate: dict) -> CandidateFeatures:
    profile = candidate["profile"]
    signals = candidate.get("redrob_signals", {})
    career_text = _full_career_text(candidate)
    yoe = profile.get("years_of_experience", 0)

    # --- Consulting-only check ---
    companies = {profile.get("current_company", "").lower()}
    companies |= {j.get("company", "").lower() for j in candidate.get("career_history", [])}
    all_consulting = len(companies) > 0 and all(
        any(firm in c for firm in CONSULTING_FIRMS) for c in companies if c
    )

    # --- Pure research, no production deployment ---
    research_signals = any(
        kw in career_text for kw in ["research scientist", "research lab", "academic", "phd researcher"]
    )
    production_signals = any(
        kw in career_text for kw in ["deployed", "production", "shipped", "real users", "scaled to"]
    )
    is_pure_research = research_signals and not production_signals

    # --- Recent LangChain-only AI experience ---
    has_langchain_wrapper = any(kw in career_text for kw in LANGCHAIN_WRAPPER_KEYWORDS)
    has_deep_ml_history = yoe >= 5 and any(
        kw in career_text for kw in (RETRIEVAL_KEYWORDS | VECTOR_DB_KEYWORDS | EVAL_FRAMEWORK_KEYWORDS)
    )
    is_recent_langchain_only = has_langchain_wrapper and not has_deep_ml_history

    # --- Stale tech lead: senior person, no recent hands-on signal ---
    is_tech_lead_title = any(t in profile.get("current_title", "").lower() for t in ["tech lead", "architect", "vp", "director"])
    last_active = _parse_date(signals.get("last_active_date"))
    last_active_days = (TODAY - last_active).days if last_active else 9999
    is_stale_tech_lead = is_tech_lead_title and last_active_days > 365

    # --- CV/Speech/robotics-only without NLP/IR ---
    has_cv_speech_robotics = any(kw in career_text for kw in CV_SPEECH_ROBOTICS_KEYWORDS)
    has_nlp_ir = any(kw in career_text for kw in NLP_IR_KEYWORDS)
    is_cv_only = has_cv_speech_robotics and not has_nlp_ir

    # --- Location ---
    loc_lower = profile.get("location", "").lower()
    in_preferred_city = any(city in loc_lower for city in TIER1_CITIES_PREFERRED)

    honeypot_flags = detect_honeypot(candidate)

    return CandidateFeatures(
        candidate_id=candidate["candidate_id"],
        years_of_experience=yoe,
        current_title=profile.get("current_title", ""),
        current_company=profile.get("current_company", ""),
        location=profile.get("location", ""),
        country=profile.get("country", ""),
        is_consulting_only=all_consulting,
        is_pure_research_no_production=is_pure_research,
        is_recent_langchain_only=is_recent_langchain_only,
        is_stale_tech_lead=is_stale_tech_lead,
        is_cv_speech_robotics_only=is_cv_only,
        in_preferred_city=in_preferred_city,
        willing_to_relocate=signals.get("willing_to_relocate", False),
        honeypot_flags=honeypot_flags,
        is_likely_honeypot=len(honeypot_flags) > 0,
        last_active_days_ago=last_active_days,
        recruiter_response_rate=signals.get("recruiter_response_rate", 0.0),
        interview_completion_rate=signals.get("interview_completion_rate", 0.0),
        open_to_work=signals.get("open_to_work_flag", False),
        notice_period_days=signals.get("notice_period_days", 999),
        github_activity_score=signals.get("github_activity_score", -1.0),
        profile_completeness=signals.get("profile_completeness_score", 0.0),
        full_text=career_text,
    )
