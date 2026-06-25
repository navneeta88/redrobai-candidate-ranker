# System Architecture Note

**Project:** Intelligent Candidate Discovery & Ranking Engine
**Challenge:** India Runs — Track 1, Data & AI Challenge (Redrob AI)

## 1. Problem framing

Given one job description and 100,000 candidate profiles, produce a ranked
top-100 shortlist with a short, explainable reason for every rank. The
explicit trap built into this challenge: the "obvious" solution (keyword
match between JD and candidate skills) is wrong by design — the dataset
contains candidates who keyword-match perfectly but are disqualified by
the JD's actual narrative requirements (pure researchers, consulting-only
careers, LangChain-wrapper-only AI experience, stale tech leads), plus a
small number of honeypot profiles with internally impossible data.

This shaped every architectural decision below: **the system cannot be a
single similarity score.** It has to separate "does this look like a
semantic match" from "does this actually satisfy what the JD is asking
for" from "is this candidate even real and available."

## 2. Pipeline overview

```
job_description.docx (read once, by us, by hand)
        |
        v
 jd_parser.py  ──────────────►  JDRequirements (keyword sets, experience
                                  band, preferred cities, semantic text)
                                                |
candidates.jsonl (100,000 records)              |
        |                                       |
        v                                       |
 candidate_features.py                          |
   - parses profile/career/education/skills     |
   - flags 6 rule-based concerns                |
   - detects honeypots (internal consistency)   |
        |                                       |
        v                                       v
              scorer.py
   - TF-IDF cosine similarity (semantic fit)
   - rule-based multiplier (concerns/boosts from JD)
   - behavioral availability multiplier (activity, response rate)
   - composite score = semantic x rules x availability
   - builds reasoning text from the SAME variables used to score
        |
        v
 rank.py (orchestrator)
   - excludes likely honeypots
   - sorts by (-score, candidate_id)
   - writes submission.csv (top 100)
```

Four modules, each independently testable:

| Module | Responsibility |
|---|---|
| `jd_parser.py` | Encodes what the JD actually requires, as structured data |
| `candidate_features.py` | Turns raw JSON into structured signals + honeypot flags |
| `scorer.py` | Combines signals into one score + matching explanation |
| `rank.py` | Orchestrates the above, end to end, into `submission.csv` |

## 3. Why three separate signal types, combined multiplicatively

**Semantic fit (TF-IDF cosine similarity).** Catches candidates whose
career history demonstrates the right kind of work even without literal
keyword overlap with the JD.

**Rule-based fit.** Hard-coded checks straight from JD language for
things TF-IDF similarity cannot reliably catch: a candidate can describe
"recommendation systems" in glowing terms while their entire career was
at a single consulting firm doing client-staffed work — semantically
similar, substantively disqualified. Six rule-based flags exist:
consulting-only career, pure research with no production deployment,
recent LangChain-wrapper-only AI experience, stale senior/lead title with
no recent platform activity, CV/speech/robotics-only background without
NLP/IR exposure, and experience outside the JD's flexible 5–9 year band.

**Behavioral availability.** A perfect on-paper match who hasn't logged
in for six months and never responds to recruiters is not, practically,
available. This is multiplied in last, separately from job fit, because
unavailability is a different kind of problem than skill mismatch and
deserves a separate, explainable line in the reasoning.

These three combine **multiplicatively**, not additively:

```
composite_score = semantic_similarity x rule_multiplier x availability_multiplier
```

A multiplicative model means one severe rule violation (e.g. ×0.35 for
pure research with no production deployment) cannot be fully offset by a
strong semantic score. This mirrors the JD's own language — phrases like
"we will not move forward" read as a near-veto, not a minor deduction. An
additive model would let high semantic similarity paper over a hard
disqualifier, which is exactly the trap the challenge is testing for.

## 4. Honeypot handling

`candidate_features.py` runs three internal-consistency checks on every
candidate:

1. Total claimed career-history duration significantly exceeds total
   years_of_experience
2. "Expert" proficiency claimed in a skill used for ~0 months
3. A single role's duration alone exceeds total claimed years of experience

A fourth check was attempted and discarded — see Section 6.

Candidates flagged by any of these are **excluded outright** from the
final top-100, not merely penalized. Per the spec, ranking more than 10%
honeypots in the top 100 disqualifies the entire submission; a missed
honeypot is far more costly than a mistakenly-excluded real candidate, so
the system is deliberately conservative in the costly direction.

## 5. Compute-budget decision: TF-IDF over sentence-transformers

The spec requires the full ranking run to complete in under 5 minutes, on
CPU only, with no network access, for the full 100,000-candidate pool.

`sentence-transformers` was not available in our build/test environment —
no network access meant no way to download model weights. Rather than
assume it would fit the budget if it were available, we measured the
alternative directly: TF-IDF vectorization (`scikit-learn`,
`TfidfVectorizer`, 5,000 features, unigrams+bigrams) fits and transforms
the full 100K-candidate corpus in ~30 seconds, and cosine similarity
against the JD vector takes under 5 seconds. Combined with feature
extraction (~30s) and scoring (under 1s), end-to-end runtime measured
**60–120 seconds** across two different machines (our build sandbox and
a Windows laptop), comfortably inside the 300-second budget.

We view this as a deliberate, documented v1 choice rather than a
limitation we're unaware of. A production v2 would very likely use a
small sentence-embedding model (e.g. a distilled MiniLM-class model) for
materially better semantic depth on the longer career-history text,
provided the model can be bundled/cached so it doesn't need a network
call at scoring time, and the added latency is re-measured against the
budget rather than assumed safe.

## 6. What we tried and discarded

**Education-vs-career timing honeypot check.** An early version of
`detect_honeypot()` flagged candidates whose listed degree finished after
their earliest job's start date, on the theory that this is an impossible
timeline. Tested against a 2,000-candidate sample, this single check
produced an **18% flag rate** — two orders of magnitude above the
dataset's expected ~0.08% honeypot rate. Manually inspecting a flagged
candidate showed the cause: completing a part-time M.Tech in 2022 while
working as an accountant since 2014 is a completely normal profile, not
an impossible one. The check was removed entirely. After removal, the
remaining three checks produced a 0.04–0.045% flag rate on larger
samples, the right order of magnitude.

**Reasoning text mislabeling a positive as a concern.** The first working
version of the reasoning generator put the preferred-city boost into the
same list as actual rule violations, all under a blanket "Concerns:"
label — so a candidate's only "concern" might literally be that they live
in Pune, which matches the JD's location preference. Caught by manually
reading the actual top-10 output after the first full run, not by a unit
test. Fixed by splitting the rule-based reasons into separate
concerns/positives lists, each labeled correctly in the final text.

Both issues are also documented in the corresponding git commit messages,
since they reflect real iteration on the actual dataset rather than
hypothetical edge cases.

## 7. Known limitations

- TF-IDF captures lexical/n-gram overlap better than it captures deep
  semantic paraphrase; a candidate describing equivalent work in very
  different vocabulary from the JD may score lower than they should.
- Rule-based multiplier values were chosen by reasoning about the
  relative severity of JD language, not fit to any ground-truth labels
  (none exist for this task). They are a defensible starting point, not
  a tuned result.
- Honeypot detection is intentionally conservative and likely has false
  negatives (real honeypots we don't catch) in exchange for very low
  false positives (real candidates we don't wrongly exclude).