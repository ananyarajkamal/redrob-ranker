# Redrob Hackathon — Candidate Ranker

Ranks 100,000 candidates against the "Senior AI Engineer — Founding Team" JD
and outputs the top 100 as `submission.csv`.

## Data

`candidates.jsonl` (~465 MB, 100K candidates) is the dataset provided by the
India Runs Hackathon organizers and is **not included in this repo** (it exceeds
GitHub's file size limit and is the organizers' data to distribute).

To reproduce results, download the dataset from the hackathon portal and place
it at the repo root as `candidates.jsonl` before running any of the steps below.

## Setup

```bash
pip install -r requirements.txt
```

## Reproduce (two steps)

**Step 1 — offline precompute (not timed, needs network once to download the
embedding model; not part of the 5-minute budget):**

```bash
python precompute_embeddings.py --candidates ./candidates.jsonl --out_dir ./artifacts
```

This embeds the JD and all candidates with a local `sentence-transformers`
model (`all-MiniLM-L6-v2`, ~90MB, downloaded once from HuggingFace) and caches
the vectors + candidate metadata to `./artifacts/`. Takes a few minutes on CPU
for 100K candidates.

If sentence-transformers / HuggingFace Hub is unreachable in your environment
(e.g. fully offline grading sandbox), use the pure scikit-learn fallback —
no network or model download required at all:

```bash
python precompute_embeddings.py --candidates ./candidates.jsonl --out_dir ./artifacts --engine tfidf
```

**Step 2 — timed ranking step (CPU only, no network, <5 min, <16GB RAM):**

```bash
python rank.py --candidates ./candidates.jsonl --artifacts ./artifacts --out ./submission.csv
```

This only loads the cached artifacts and scores/ranks — no embedding model is
loaded or called here, so it's fast and fully offline. On the full 100K
candidate set this completes in well under a minute on a single CPU core.

**Step 3 — validate:**

```bash
python validate_submission.py ./submission.csv
```

## How it works

1. **Semantic similarity**: JD and each candidate (headline, summary, career
   history, skills) are embedded and compared via cosine similarity. This
   captures overall narrative fit.
2. **Structured JD facet checklist**: the JD is also broken into individually
   weighted, individually checkable requirements (production embeddings and
   retrieval, vector database and hybrid search, ranking evaluation
   frameworks, strong Python, LLM fine-tuning, learning to rank, HR-tech
   background, distributed systems, open-source contributions). Each
   candidate is checked against each facet directly, using their own career
   text and skills, not the JD's text. Missing a must-have facet applies a
   direct penalty. This is what lets the ranker explain exactly which JD
   requirements a candidate does and does not satisfy, instead of relying on
   one opaque similarity number, and it catches keyword-stuffed profiles that
   an embedding-only approach can be fooled by.
3. Final semantic score is a blend of embedding similarity and facet score.
4. **Hard penalty gate**: encodes the JD's explicit "who we do NOT want"
   section, described below.
5. **Experience-years fit**, **location fit**, **honeypot filter**, and
   **behavioral multiplier** as described below.
6. Final score = combined_semantic_score x hard_penalty x facet_penalty x
   experience_fit x location_fit x behavioral_multiplier. Reasoning text is
   generated from the same real fields that drove the score, including which
   facets matched and which were missing, no LLM call, no hallucinated
   skills.

Details on each component:

- **Hard penalty gate**: pure-research-only backgrounds with no production
  deployment, 100%-consulting-firm careers (TCS/Infosys/Wipro/Accenture/
  Cognizant/Capgemini) with no product-company experience, CV/speech/
  robotics-only backgrounds with no NLP/IR exposure, title-hopping (under 18
  months average tenure), framework-tutorial-only skills pattern with no
  retrieval evidence, and off-track current titles (Marketing/HR/Sales/Ops
  Manager, Customer Support, Content Writer) regardless of listed skills.
- **Experience-years fit**: soft curve centered on the JD's 5 to 9 year band.
- **Location fit**: Pune/Noida preferred, Hyderabad/Mumbai/Delhi NCR
  acceptable, rest of India and international lower but not excluded.
- **Honeypot filter**: hard-excludes profiles with internal inconsistencies,
  for example "expert" skill proficiency with 0 months used, years of
  experience far exceeding the sum of career-history durations, or
  overlapping/duplicate "current" roles.
- **Behavioral multiplier**: built from the 23 redrob_signals fields,
  activity recency, recruiter response rate, interview completion rate,
  profile completeness, so a great-on-paper but inactive or unresponsive
  candidate is down-weighted.

## Proof the logic actually works: adversarial test suite

`test_adversarial.py` builds 10 synthetic candidates specifically designed to
trip up a naive keyword-matching ranker: a keyword-stuffed profile with an
unrelated job title, a plain-language candidate who is a genuine fit but uses
no jargon, a pure-consulting career, a research-only background, a
CV/robotics-only background, a job-hopper, two honeypots with internally
impossible data, and a strong-fit candidate who has gone inactive. Run it
with:

```bash
python test_adversarial.py
```

All 10 cases pass their expected bucket (top, bottom, or excluded) against
this ranker's actual output, and the script prints the full score breakdown
for each one. This script uses a TF-IDF stand-in for the semantic similarity
score so it runs standalone with no network or model download. The real
pipeline uses sentence-transformers instead, which handles plain-language
paraphrases better than TF-IDF term overlap does.

## Files

- `common.py` — shared scoring and feature logic: JD facet checklist,
  penalty rules, honeypot detection, behavioral multiplier, reasoning
  generation
- `precompute_embeddings.py` — offline embedding step
- `rank.py` — timed ranking step, produces `submission.csv`
- `test_adversarial.py` — adversarial synthetic test suite proving the
  scoring logic handles known trap patterns correctly
- `validate_submission.py` — format validator (provided by organizers)
- `requirements.txt`

## Compute profile

Tested on 100,000 real candidates:
- Precompute (offline): ~95s with the scikit-learn TF-IDF fallback on 1 CPU
  core / 4GB RAM (sentence-transformers is slower but still well within a
  reasonable offline window, and produces higher quality semantic matches)
- Ranking step (timed): ~17s on 1 CPU core / 4GB RAM — well under the 5-minute
  budget even on modest hardware
- 0 honeypots surfaced in top 100 on this run