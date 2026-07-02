"""
Shared logic for the Redrob candidate ranker.
Used by both precompute_embeddings.py (offline) and rank.py (timed step).
"""
import json
import gzip
import re
from datetime import date, datetime

# ---------------------------------------------------------------------------
# JD-derived constants (encoded from job_description.docx)
# ---------------------------------------------------------------------------

JD_TEXT = """
Senior AI Engineer, Founding Team, Redrob AI. Series A AI-native talent
intelligence platform. Own the intelligence layer: ranking, retrieval, and
matching systems for candidate-JD matching. Production experience with
embeddings-based retrieval systems (sentence-transformers, OpenAI embeddings,
BGE, E5) deployed to real users, handling embedding drift, index refresh,
retrieval-quality regression in production. Production experience with vector
databases or hybrid search infrastructure: Pinecone, Weaviate, Qdrant, Milvus,
OpenSearch, Elasticsearch, FAISS. Strong Python and code quality. Hands-on
experience designing evaluation frameworks for ranking systems: NDCG, MRR,
MAP, offline-to-online correlation, A/B testing. LLM fine-tuning (LoRA, QLoRA,
PEFT), learning-to-rank models, HR-tech or marketplace product background,
distributed systems, large-scale inference optimization, open-source
contributions are a plus. Ideal candidate: 6-8 years total experience, 4-5
years in applied ML/AI roles at product companies, has shipped an end-to-end
ranking, search, or recommendation system to real users at meaningful scale,
strong opinions on hybrid retrieval, offline vs online evaluation, and when to
fine-tune vs prompt an LLM, defensible with reference to systems actually
built.
"""

CONSULTING_FIRMS = {
    "tcs", "tata consultancy services", "infosys", "wipro", "accenture",
    "cognizant", "capgemini",
}

PREFERRED_LOCATIONS = {"pune", "noida"}
ACCEPTABLE_LOCATIONS = {"hyderabad", "mumbai", "delhi", "new delhi", "gurgaon", "gurugram", "delhi ncr"}

RETRIEVAL_KEYWORDS = [
    "embedding", "retrieval", "vector search", "vector database", "faiss",
    "pinecone", "weaviate", "qdrant", "milvus", "elasticsearch", "opensearch",
    "ranking system", "recommendation system", "recommender system",
    "search relevance", "semantic search", "hybrid search", "re-rank",
    "reranking", "ndcg", "mrr", "learning to rank", "ann index",
    "approximate nearest neighbor",
]

RESEARCH_ONLY_KEYWORDS = [
    "research scientist", "phd researcher", "academic lab", "research fellow",
    "postdoc", "research intern", "publication-only",
]

CV_SPEECH_ROBOTICS_KEYWORDS = [
    "computer vision", "speech recognition", "robotics", "autonomous driving",
    "lidar", "slam", "image classification", "object detection",
]

NLP_IR_KEYWORDS = [
    "nlp", "natural language processing", "information retrieval", "llm",
    "language model", "text classification", "embeddings", "transformer",
    "bert", "gpt", "search", "ranking",
]

FRAMEWORK_TUTORIAL_KEYWORDS = [
    "langchain tutorial", "how i used", "demo project", "toy project",
    "followed a tutorial", "built a demo",
]

RECENT_LANGCHAIN_ONLY_KEYWORDS = ["langchain", "openai api", "prompt engineering"]

# ---------------------------------------------------------------------------
# Structured JD facets. Instead of treating the JD as one blob to embed,
# we break it into individually weighted, individually checkable requirements.
# Each facet has: a name, a weight, a type (must_have, disqualifier,
# nice_to_have), and a list of keyword phrases used to detect evidence for it
# in the candidate's own text. This lets the ranker explain exactly which
# parts of the JD a candidate does and does not satisfy, instead of relying
# on one opaque similarity number.
# ---------------------------------------------------------------------------

JD_FACETS = [
    {
        "name": "production_embeddings_retrieval",
        "type": "must_have",
        "weight": 0.22,
        "keywords": [
            "sentence-transformers", "sentence transformers", "openai embeddings",
            "bge", "e5 embedding", "embedding drift", "index refresh",
            "retrieval quality", "production embedding", "embedding pipeline",
            "embedding", "embeddings",
        ],
    },
    {
        "name": "vector_db_hybrid_search",
        "type": "must_have",
        "weight": 0.18,
        "keywords": [
            "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
            "elasticsearch", "faiss", "vector database", "hybrid search",
            "vector search",
        ],
    },
    {
        "name": "ranking_eval_frameworks",
        "type": "must_have",
        "weight": 0.18,
        "keywords": [
            "ndcg", "mrr", "map@", "mean average precision", "offline evaluation",
            "online evaluation", "a/b test", "ab test", "evaluation framework",
            "offline-to-online",
        ],
    },
    {
        "name": "strong_python_production_code",
        "type": "must_have",
        "weight": 0.10,
        "keywords": [
            "python", "production code", "code review", "unit test", "ci/cd",
        ],
    },
    {
        "name": "llm_finetuning",
        "type": "nice_to_have",
        "weight": 0.08,
        "keywords": ["lora", "qlora", "peft", "fine-tuning", "fine-tune"],
    },
    {
        "name": "learning_to_rank",
        "type": "nice_to_have",
        "weight": 0.08,
        "keywords": ["learning to rank", "ltr", "xgboost ranking", "neural ranking"],
    },
    {
        "name": "hrtech_marketplace_background",
        "type": "nice_to_have",
        "weight": 0.06,
        "keywords": ["hr-tech", "hr tech", "recruiting platform", "marketplace", "talent platform"],
    },
    {
        "name": "distributed_systems_scale",
        "type": "nice_to_have",
        "weight": 0.06,
        "keywords": ["distributed systems", "large-scale inference", "kafka", "spark", "high throughput"],
    },
    {
        "name": "open_source_external_validation",
        "type": "nice_to_have",
        "weight": 0.04,
        "keywords": ["open source", "github contributions", "published a paper", "conference talk", "blog post"],
    },
]

MUST_HAVE_MISS_PENALTY = 0.6  # multiplier applied per missed must-have facet


def score_jd_facets(c):
    """
    Checks each JD facet against the candidate's own text (not the JD's text),
    returning a dict of facet_name -> (hit_count, contribution) plus a combined
    facet_score in 0-1 and a list of missed must-have facet names.
    Independent of and complementary to the semantic embedding similarity score.
    """
    blob = career_blob(c)
    skills = c.get("skills", []) or []
    skill_blob = " ".join(_safe_lower(s.get("name", "")) for s in skills)
    full_blob = blob + " " + skill_blob

    facet_hits = {}
    weighted_sum = 0.0
    total_weight = 0.0
    missed_must_haves = []

    for facet in JD_FACETS:
        hits = sum(1 for kw in facet["keywords"] if kw in full_blob)
        satisfied = hits > 0
        facet_hits[facet["name"]] = hits
        total_weight += facet["weight"]
        if satisfied:
            weighted_sum += facet["weight"]
        elif facet["type"] == "must_have":
            missed_must_haves.append(facet["name"])

    facet_score = weighted_sum / total_weight if total_weight > 0 else 0.0
    return {
        "facet_score": facet_score,
        "facet_hits": facet_hits,
        "missed_must_haves": missed_must_haves,
    }


def facet_penalty(missed_must_haves):
    """Each missed must-have facet multiplies the score down."""
    penalty = 1.0
    for _ in missed_must_haves:
        penalty *= MUST_HAVE_MISS_PENALTY
    return penalty


def load_candidates(path):
    """Yields parsed candidate dicts. Handles both .jsonl and .jsonl.gz."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _safe_lower(x):
    return (x or "").lower()


def candidate_text(c):
    """Build the text blob used for embedding. Weights recent roles higher by repetition."""
    profile = c.get("profile", {})
    parts = [
        profile.get("headline", ""),
        profile.get("summary", ""),
        profile.get("current_title", ""),
        profile.get("current_title", ""),  # repeat current title -> extra weight
    ]
    career = c.get("career_history", []) or []
    # sort most-recent first (is_current, then start_date desc)
    def sort_key(entry):
        return (not entry.get("is_current", False), entry.get("start_date", "") or "")
    career_sorted = sorted(career, key=sort_key)
    for i, entry in enumerate(career_sorted[:5]):
        weight = 2 if i == 0 else 1  # most recent role weighted x2
        text = f"{entry.get('title','')} at {entry.get('company','')}: {entry.get('description','')}"
        parts.extend([text] * weight)

    skills = c.get("skills", []) or []
    # weight skills by endorsements + duration, so lazily-listed skills contribute less
    skill_terms = []
    for s in sorted(skills, key=lambda s: (s.get("endorsements", 0), s.get("duration_months", 0)), reverse=True)[:15]:
        if s.get("duration_months", 0) >= 6:  # only include skills with real usage evidence
            skill_terms.append(s.get("name", ""))
    parts.append(" ".join(skill_terms))

    return " ".join(p for p in parts if p)


def years_experience_fit(years):
    """Soft curve centered on 5-9 yrs. Returns multiplier 0.55-1.0."""
    if years is None:
        return 0.7
    if 5 <= years <= 9:
        return 1.0
    if years < 5:
        gap = 5 - years
        return max(0.55, 1.0 - 0.12 * gap)
    gap = years - 9
    return max(0.6, 1.0 - 0.08 * gap)


def location_fit(location, country):
    loc = _safe_lower(location)
    country = _safe_lower(country)
    if country and country != "india":
        return 0.75  # outside India: case-by-case, no visa sponsorship
    for p in PREFERRED_LOCATIONS:
        if p in loc:
            return 1.0
    for a in ACCEPTABLE_LOCATIONS:
        if a in loc:
            return 0.9
    return 0.8


def is_pure_consulting_career(career):
    if not career:
        return False
    companies = [_safe_lower(e.get("company", "")) for e in career]
    return all(any(cf in comp for cf in CONSULTING_FIRMS) for comp in companies) and len(companies) > 0


def avg_tenure_months(career):
    durations = [e.get("duration_months", 0) for e in career if e.get("duration_months")]
    if not durations:
        return None
    return sum(durations) / len(durations)


def career_blob(c):
    career = c.get("career_history", []) or []
    parts = [f"{e.get('title','')} {e.get('description','')}" for e in career]
    parts.append(c.get("profile", {}).get("summary", ""))
    parts.append(c.get("profile", {}).get("headline", ""))
    return _safe_lower(" ".join(parts))


def has_retrieval_evidence(c):
    blob = career_blob(c)
    return sum(1 for k in RETRIEVAL_KEYWORDS if k in blob)


def hard_penalty_gate(c):
    """
    Returns a multiplier 0.0-1.0 encoding the JD's explicit "do NOT want" list.
    Multiple violations stack multiplicatively.
    """
    penalty = 1.0
    career = c.get("career_history", []) or []
    blob = career_blob(c)
    profile = c.get("profile", {})
    title = _safe_lower(profile.get("current_title", ""))

    # Pure research-only background, no production deployment
    research_hits = sum(1 for k in RESEARCH_ONLY_KEYWORDS if k in blob)
    has_production_evidence = any(w in blob for w in ["deployed", "production", "shipped", "launched", "scale"])
    if research_hits > 0 and not has_production_evidence:
        penalty *= 0.2

    # 100% consulting-firm career with no product company experience
    if is_pure_consulting_career(career):
        penalty *= 0.25

    # CV/speech/robotics-only, no NLP/IR exposure
    cv_hits = sum(1 for k in CV_SPEECH_ROBOTICS_KEYWORDS if k in blob)
    nlp_hits = sum(1 for k in NLP_IR_KEYWORDS if k in blob)
    if cv_hits >= 2 and nlp_hits == 0:
        penalty *= 0.3

    # Title-hopping: avg tenure well under 18 months across a multi-role career
    tenure = avg_tenure_months(career)
    if tenure is not None and len(career) >= 3 and tenure < 18:
        penalty *= 0.5

    # Framework-tutorial-only signal: keywords present but no retrieval evidence
    tut_hits = sum(1 for k in FRAMEWORK_TUTORIAL_KEYWORDS if k in blob)
    if tut_hits > 0 and has_retrieval_evidence(c) == 0:
        penalty *= 0.4

    # Recent-only LangChain/OpenAI-wrapper experience, no pre-LLM-era production ML
    skills = c.get("skills", []) or []
    skill_names = {_safe_lower(s.get("name", "")) for s in skills}
    langchain_only = skill_names & set(RECENT_LANGCHAIN_ONLY_KEYWORDS)
    pre_llm_evidence = any(w in blob for w in ["spark", "recommendation", "search", "ranking", "retrieval", "ml pipeline", "machine learning"])
    if langchain_only and not pre_llm_evidence and has_retrieval_evidence(c) == 0:
        penalty *= 0.35

    # Not senior-tech-lead-who-doesn't-code: penalize pure management/architecture titles w/ no recent hands-on
    if any(t in title for t in ["marketing manager", "hr manager", "sales manager", "operations manager", "customer support", "content writer"]):
        penalty *= 0.05  # clearly off-track roles regardless of skills listed

    return max(penalty, 0.02)


def is_honeypot(c):
    """Detect subtly impossible profiles."""
    career = c.get("career_history", []) or []
    skills = c.get("skills", []) or []
    profile = c.get("profile", {})

    # expert proficiency with 0 (or near-0) duration_months used
    for s in skills:
        if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0:
            return True

    # years_of_experience wildly inconsistent with sum of career durations
    yoe = profile.get("years_of_experience")
    total_months = sum(e.get("duration_months", 0) for e in career)
    if yoe is not None and total_months > 0:
        implied_years = total_months / 12.0
        if yoe > implied_years * 2.5 and yoe - implied_years > 3:
            return True

    # overlapping career_history date ranges (more than one "is_current" or overlapping spans)
    current_count = sum(1 for e in career if e.get("is_current"))
    if current_count > 1:
        return True

    parsed = []
    for e in career:
        try:
            sd = datetime.strptime(e["start_date"][:10], "%Y-%m-%d").date()
        except Exception:
            continue
        ed_raw = e.get("end_date")
        ed = date.today() if not ed_raw else datetime.strptime(ed_raw[:10], "%Y-%m-%d").date()
        parsed.append((sd, ed))
    parsed.sort()
    for i in range(len(parsed) - 1):
        if parsed[i][1] > parsed[i + 1][0]:
            # allow small overlap (<45 days, common in real data for transitions)
            if (parsed[i][1] - parsed[i + 1][0]).days > 45:
                return True

    return False


def behavioral_multiplier(c):
    """0.4-1.0 based on the 23 redrob_signals."""
    sig = c.get("redrob_signals", {}) or {}
    mult = 1.0

    # recency of activity
    last_active = sig.get("last_active_date")
    if last_active:
        try:
            la = datetime.strptime(last_active[:10], "%Y-%m-%d").date()
            days_inactive = (date.today() - la).days
            if days_inactive > 180:
                mult *= 0.5
            elif days_inactive > 90:
                mult *= 0.75
            elif days_inactive > 30:
                mult *= 0.9
        except Exception:
            pass

    resp_rate = sig.get("recruiter_response_rate")
    if resp_rate is not None:
        mult *= (0.6 + 0.4 * resp_rate)  # 0.6-1.0

    interview_rate = sig.get("interview_completion_rate")
    if interview_rate is not None:
        mult *= (0.8 + 0.2 * interview_rate)  # 0.8-1.0

    open_to_work = sig.get("open_to_work_flag")
    if open_to_work is False:
        mult *= 0.85

    completeness = sig.get("profile_completeness_score")
    if completeness is not None:
        mult *= (0.85 + 0.15 * (completeness / 100.0))  # 0.85-1.0

    return max(mult, 0.3)


def build_reasoning(c, sim_score, penalty, behav_mult, retrieval_hits, facet_result=None):
    profile = c.get("profile", {})
    yoe = profile.get("years_of_experience")
    title = profile.get("current_title", "")
    company = profile.get("current_company", "")
    location = profile.get("location", "")
    sig = c.get("redrob_signals", {}) or {}
    resp_rate = sig.get("recruiter_response_rate")

    bits = [f"{title} with {yoe} yrs, currently at {company} ({location})."]

    if facet_result:
        matched = [name for name, hits in facet_result["facet_hits"].items() if hits > 0]
        must_have_matched = [
            f["name"] for f in JD_FACETS
            if f["type"] == "must_have" and f["name"] in matched
        ]
        if must_have_matched:
            readable = ", ".join(n.replace("_", " ") for n in must_have_matched)
            bits.append(f"Matches JD must-haves: {readable}.")
        if facet_result["missed_must_haves"]:
            readable_miss = ", ".join(n.replace("_", " ") for n in facet_result["missed_must_haves"])
            bits.append(f"Missing evidence for: {readable_miss}.")
    elif retrieval_hits > 0:
        bits.append(f"Career history shows {retrieval_hits} direct signal(s) of ranking/retrieval/search work.")
    else:
        bits.append("Limited direct ranking/retrieval evidence in career history.")

    if penalty < 0.5:
        bits.append("Some concerns flagged against the JD's explicit disqualifiers.")
    if resp_rate is not None:
        bits.append(f"Recruiter response rate {resp_rate:.0%}.")
    return " ".join(bits)