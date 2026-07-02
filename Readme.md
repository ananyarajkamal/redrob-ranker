# Candidate Ranker

Ranks 100,000 candidates against the "Senior AI Engineer - Founding Team" JD
and outputs the top N as `submission.csv`.

Also ships a local web UI: upload a candidates file, choose how many to rank,
view the results with JD facet coverage per candidate, and download the CSV.

## Data

`candidates.jsonl` (~465 MB, 100K candidates) is the dataset provided by the
hackathon organizers and is **not included in this repo** (it exceeds GitHub's
file size limit and is the organizers' data to distribute).

**Download the dataset here:**
[candidates.jsonl on Google Drive](https://drive.google.com/file/d/1MfD47XvVdRKBGRAyzGOxDCEf2ve96Jjo/view)

Place it at the repo root as `candidates.jsonl` before running any of the
steps below.

## Setup

> **Before running anything:** `candidates.jsonl` must be in the repo root.
> Download it from the link above. It is not included because it is 465 MB
> and belongs to the hackathon organizers.

```bash
pip install -r requirements.txt
```

## Reproduce (two steps)

**Step 1 - offline precompute (not timed, needs network once to download the
embedding model; not part of the 5-minute budget):**

```bash
python precompute_embeddings.py --candidates ./candidates.jsonl --out_dir ./artifacts
```

This embeds the JD and all candidates with a local `sentence-transformers`
model (`all-MiniLM-L6-v2`, ~90 MB, downloaded once from HuggingFace) and
caches the vectors and candidate metadata to `./artifacts/`. Takes a few
minutes on CPU for 100K candidates. The `artifacts/` folder is created
automatically.

If sentence-transformers or HuggingFace Hub is unreachable (fully offline
grading sandbox), use the scikit-learn fallback, no network or model download
required:

```bash
python precompute_embeddings.py --candidates ./candidates.jsonl --out_dir ./artifacts --engine tfidf
```

**Step 2 - timed ranking step (CPU only, no network, under 5 min, under 16 GB RAM):**

```bash
python rank.py --candidates ./candidates.jsonl --artifacts ./artifacts --out ./submission.csv
```

Only loads cached artifacts and scores/ranks. No embedding model is loaded or
called here, so it is fast and fully offline. On the full 100K candidate set
this completes in under 20 seconds on a single CPU core.

**Step 3 - validate:**

```bash
python validate_submission.py ./submission.csv
```

## Web UI (optional)

To explore results interactively, start the local web server:

```bash
python -m uvicorn app:app --port 8000
```

Then open `http://localhost:8000` in your browser.

Upload a `candidates.jsonl` file, choose how many candidates to return (10,
50, 100, 500, 1000, or any number), and click Run ranking. The server uses
the precomputed artifacts if they exist, so ranking is as fast as the
command-line step. Results show the ranked table with scores and reasoning.
Click any row to see the full JD facet coverage for that candidate. Export
the ranked list as a CSV from the same page.

## How it works

1. **Semantic similarity**: JD and each candidate (headline, summary, career
   history, skills) are embedded and compared via cosine similarity. This
   captures overall narrative fit without requiring the candidate to use the
   JD's exact words.
2. **Structured JD facet checklist**: the JD is broken into 9 individually
   weighted, individually checkable requirements (production embeddings and
   retrieval, vector database and hybrid search, ranking evaluation
   frameworks, strong Python, LLM fine-tuning, learning to rank, HR-tech
   background, distributed systems, open-source contributions). Each
   candidate is checked against each facet using their own career text and
   skills, not the JD's text. Missing a must-have facet applies a direct
   penalty. This is what lets the ranker explain exactly which JD requirements
   a candidate does and does not satisfy, and it catches keyword-stuffed
   profiles that an embedding-only approach can be fooled by.
3. Final semantic score is a blend of embedding similarity and facet score:
   `combined_semantic = 0.55 x embedding_similarity + 0.45 x facet_score`
4. **Hard penalty gate**: encodes the JD's explicit "who we do NOT want"
   section (details below).
5. **Experience-years fit**, **location fit**, **honeypot filter**, and
   **behavioral multiplier** (details below).
6. `final_score = combined_semantic x hard_penalty x facet_penalty x
   experience_fit x location_fit x behavioral_multiplier`. Reasoning text is
   generated from the same real fields that drove the score, including which
   facets matched and which were missing. No LLM is called, no skills are
   invented.

Details on each component:

- **Hard penalty gate**: pure-research-only backgrounds with no production
  deployment, 100% consulting-firm careers (TCS, Infosys, Wipro, Accenture,
  Cognizant, Capgemini) with no product-company experience, CV/speech/
  robotics-only backgrounds with no NLP/IR exposure, title-hopping (under 18
  months average tenure), framework-tutorial-only skills with no retrieval
  evidence, and off-track current titles (Marketing, HR, Sales, Ops Manager,
  Customer Support, Content Writer) regardless of listed skills.
- **Experience-years fit**: soft curve centered on the JD's 5 to 9 year band.
- **Location fit**: Pune/Noida preferred, Hyderabad/Mumbai/Delhi NCR
  acceptable, rest of India and international lower but not excluded.
- **Honeypot filter**: hard-excludes profiles with internal inconsistencies,
  for example "expert" skill proficiency with 0 months used, years of
  experience far exceeding the sum of career history durations, or
  overlapping/duplicate current roles.
- **Behavioral multiplier**: built from the 23 redrob_signals fields.
  Activity recency, recruiter response rate, interview completion rate, and
  profile completeness all feed in, so a great-on-paper but inactive or
  unresponsive candidate is down-weighted.

## Why NumPy matrix multiply and not FAISS

At 100K candidates, a full matrix multiply (`jd_emb @ cand_emb.T`) runs in
under a second on CPU and produces exact cosine similarities across the entire
set. An approximate nearest-neighbour index (FAISS, ScaNN) would add build
time, index size, and approximation error with no speed benefit at this scale.
The simpler approach is faster and more accurate here.

## Adversarial test suite

`test_adversarial.py` builds 10 synthetic candidates specifically designed to
trip up a naive keyword-matching ranker: a keyword-stuffed profile with an
unrelated job title, a plain-language candidate who is a genuine fit but uses
no jargon, a pure-consulting career, a research-only background, a
CV/robotics-only background, a job-hopper, two honeypots with internally
impossible data, and a strong-fit candidate who has gone inactive.

```bash
python test_adversarial.py
```

All 10 cases pass their expected bucket (top, bottom, or excluded) against the
ranker's actual output, and the script prints the full score breakdown for each
one. This script uses a TF-IDF stand-in for the semantic similarity score so
it runs standalone with no network or model download. The real pipeline uses
sentence-transformers, which handles plain-language paraphrases better than
TF-IDF term overlap.

## Files

| File | Purpose |
|------|---------|
| `common.py` | Shared scoring and feature logic: JD facet checklist, penalty rules, honeypot detection, behavioral multiplier, reasoning generation |
| `precompute_embeddings.py` | Offline embedding step |
| `rank.py` | Timed ranking step, produces `submission.csv` |
| `app.py` | FastAPI web server for the interactive UI |
| `static/` | Frontend (HTML, CSS, JS) for the web UI |
| `test_adversarial.py` | Adversarial synthetic test suite |
| `validate_submission.py` | Format validator (provided by organizers) |
| `requirements.txt` | Python dependencies |

## Compute profile

Tested on 100,000 real candidates:

| Step | Time | Resources |
|------|------|-----------|
| Precompute (offline, not timed) | ~95s (TF-IDF) | 1 CPU core, 4 GB RAM |
| Ranking (timed step) | ~17s | 1 CPU core, 4 GB RAM |
| Budget | 5 min | 16 GB RAM, CPU only, no network |

0 honeypots surfaced in the top 100. 10/10 adversarial cases caught correctly.