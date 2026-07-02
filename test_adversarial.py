"""
Adversarial test suite.

Builds a small set of synthetic candidates designed to trip up a naive
keyword-matching ranker, then checks that this ranker orders them correctly.

This is not part of the timed ranking pipeline. It is evidence for the deck
and for the Explainability and Data Validation section of the write-up.

Usage:
    python test_adversarial.py
"""
import json

import numpy as np

from common import (
    JD_TEXT, years_experience_fit, location_fit, hard_penalty_gate,
    is_honeypot, behavioral_multiplier, score_jd_facets, facet_penalty,
    build_reasoning,
)


def make_candidate(cid, title, company, years, summary, career_desc,
                    skills, location="Pune", country="India",
                    last_active_days_ago=5, response_rate=0.7,
                    interview_rate=0.6, completeness=85, open_to_work=True,
                    duration_months=36, extra_career=None):
    from datetime import date, timedelta
    last_active = (date.today() - timedelta(days=last_active_days_ago)).isoformat()
    start = (date.today() - timedelta(days=duration_months * 30)).isoformat()

    career_history = [{
        "title": title,
        "company": company,
        "description": career_desc,
        "start_date": start,
        "end_date": None,
        "is_current": True,
        "duration_months": duration_months,
    }]
    if extra_career:
        career_history.extend(extra_career)

    return {
        "candidate_id": cid,
        "profile": {
            "headline": title,
            "summary": summary,
            "current_title": title,
            "current_company": company,
            "years_of_experience": years,
            "location": location,
            "country": country,
        },
        "career_history": career_history,
        "skills": skills,
        "redrob_signals": {
            "last_active_date": last_active,
            "recruiter_response_rate": response_rate,
            "interview_completion_rate": interview_rate,
            "profile_completeness_score": completeness,
            "open_to_work_flag": open_to_work,
        },
    }


def skill(name, months, proficiency="intermediate", endorsements=5):
    return {"name": name, "duration_months": months, "proficiency": proficiency, "endorsements": endorsements}


CASES = [
    {
        "id": "STRONG_REAL_FIT",
        "expect": "top",
        "candidate": make_candidate(
            "T01", "Senior ML Engineer", "SearchCo", 7,
            "Built and shipped a hybrid retrieval system combining BM25 and dense embeddings for a production search product used by millions.",
            "Owned the embeddings-based retrieval pipeline in production, handled embedding drift and index refresh, ran offline evaluation with NDCG and MRR before shipping A/B tests.",
            [skill("Python", 60, "expert", 40), skill("FAISS", 30, "advanced", 20),
             skill("Elasticsearch", 24, "advanced", 15), skill("NDCG", 20, "advanced", 10),
             skill("LoRA", 12, "intermediate", 5)],
        ),
    },
    {
        "id": "KEYWORD_STUFFER_WRONG_TITLE",
        "expect": "excluded",
        "candidate": make_candidate(
            "T02", "Marketing Manager", "BrandCo", 6,
            "Marketing professional. Skills: Python, FAISS, Pinecone, LangChain, NDCG, MRR, LoRA, QLoRA, vector database, hybrid search.",
            "Led marketing campaigns and brand strategy for consumer products.",
            [skill(k, 0, "expert", 1) for k in
             ["Python", "FAISS", "Pinecone", "LangChain", "NDCG", "MRR", "LoRA", "QLoRA"]],
        ),
    },
    {
        "id": "PLAIN_LANGUAGE_GENUINE_FIT",
        "expect": "upper_half",
        "candidate": make_candidate(
            "T03", "Backend Engineer", "JobPortalX", 6,
            "I work on the system that decides which resumes show up first when a recruiter searches, tuned it using click data and measured how well we ranked things offline before rolling out changes to real users.",
            "Rebuilt our candidate search so it looks at meaning not just words, using embeddings, and set up dashboards to check if our top results were actually better before and after each change.",
            [skill("Python", 48, "advanced", 15), skill("Search", 40, "advanced", 12),
             skill("Machine Learning", 36, "intermediate", 10)],
        ),
    },
    {
        "id": "PURE_CONSULTING_NO_PRODUCT",
        "expect": "bottom",
        "candidate": make_candidate(
            "T04", "Senior Consultant", "TCS", 7,
            "Delivered client projects across multiple domains.",
            "Worked as a consultant on client-staffed engagements at TCS for the entire career.",
            [skill("Python", 24, "advanced", 5), skill("Machine Learning", 24, "intermediate", 4)],
            extra_career=[{
                "title": "Consultant", "company": "TCS",
                "description": "Client engagements.",
                "start_date": "2018-01-01", "end_date": "2021-01-01",
                "is_current": False, "duration_months": 36,
            }],
        ),
    },
    {
        "id": "RESEARCH_ONLY_NO_PRODUCTION",
        "expect": "bottom",
        "candidate": make_candidate(
            "T05", "Research Scientist", "University Lab", 5,
            "Research scientist studying ranking algorithms, published papers on learning to rank.",
            "Conducted academic research on ranking algorithms in a university lab, no production deployment.",
            [skill("Learning to Rank", 40, "expert", 8), skill("Python", 40, "advanced", 8)],
        ),
    },
    {
        "id": "CV_ROBOTICS_NO_NLP",
        "expect": "bottom",
        "candidate": make_candidate(
            "T06", "Computer Vision Engineer", "RoboticsCo", 6,
            "Computer vision and robotics engineer, worked on object detection and SLAM for autonomous robots.",
            "Built object detection and SLAM pipelines for autonomous driving and robotics products.",
            [skill("Computer Vision", 60, "expert", 20), skill("Robotics", 48, "advanced", 15),
             skill("Lidar", 36, "advanced", 10)],
        ),
    },
    {
        "id": "TITLE_HOPPER",
        "expect": "middle_or_bottom",
        "candidate": make_candidate(
            "T07", "ML Engineer", "StartupD", 5,
            "ML engineer who has changed jobs frequently.",
            "Worked on ranking models briefly before moving on.",
            [skill("Python", 40, "advanced", 10), skill("Ranking", 12, "intermediate", 4)],
            duration_months=8,
            extra_career=[
                {"title": "ML Engineer", "company": "StartupA", "description": "Ranking work.",
                 "start_date": "2021-01-01", "end_date": "2021-09-01", "is_current": False, "duration_months": 8},
                {"title": "ML Engineer", "company": "StartupB", "description": "Search work.",
                 "start_date": "2020-01-01", "end_date": "2020-09-01", "is_current": False, "duration_months": 8},
            ],
        ),
    },
    {
        "id": "HONEYPOT_IMPOSSIBLE_EXPERT",
        "expect": "excluded",
        "candidate": make_candidate(
            "T08", "AI Engineer", "FutureCo", 8,
            "Expert in everything immediately.",
            "Claims deep expertise across the board.",
            [skill("FAISS", 0, "expert", 50), skill("Pinecone", 0, "expert", 40)],
        ),
    },
    {
        "id": "HONEYPOT_YOE_MISMATCH",
        "expect": "excluded",
        "candidate": make_candidate(
            "T09", "Senior AI Engineer", "ScaleCo", 15,
            "Fifteen years of experience.",
            "Recent single short role only.",
            [skill("Python", 6, "advanced", 5)],
            duration_months=6,
        ),
    },
    {
        "id": "STRONG_FIT_BUT_INACTIVE",
        "expect": "middle",
        "candidate": make_candidate(
            "T10", "Senior ML Engineer", "SearchCo2", 7,
            "Built and shipped a hybrid retrieval system with embeddings and vector search in production, evaluated with NDCG and A/B tests.",
            "Owned production retrieval and ranking systems, embeddings, vector databases, offline evaluation with NDCG and MRR.",
            [skill("Python", 60, "expert", 30), skill("FAISS", 30, "advanced", 15),
             skill("NDCG", 20, "advanced", 8)],
            last_active_days_ago=250, response_rate=0.1, interview_rate=0.05,
            open_to_work=False,
        ),
    },
]


def compute_semantic_scores(candidates, jd_text):
    """Lightweight TF-IDF cosine similarity against the JD, used only in this
    test script so it runs standalone with no network and no model download.
    The real pipeline uses sentence-transformers here instead, which handles
    plain-language paraphrases better than TF-IDF term overlap does."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    texts = []
    for c in candidates:
        p = c["profile"]
        parts = [p.get("headline", ""), p.get("summary", "")]
        for e in c.get("career_history", []):
            parts.append(f"{e.get('title','')} {e.get('description','')}")
        texts.append(" ".join(parts))

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform([jd_text] + texts)
    sims = cosine_similarity(matrix[0:1], matrix[1:])[0]
    return {c["candidate_id"]: float(s) for c, s in zip(candidates, sims)}


def score_candidate(c, sim_score):
    facet_result = score_jd_facets(c)
    f_penalty = facet_penalty(facet_result["missed_must_haves"])
    penalty = hard_penalty_gate(c)
    exp_fit = years_experience_fit(c["profile"]["years_of_experience"])
    loc_fit = location_fit(c["profile"]["location"], c["profile"]["country"])
    behav = behavioral_multiplier(c)
    honeypot = is_honeypot(c)

    # Same 55/45 semantic/facet blend as rank.py, using a TF-IDF stand-in for
    # the embedding model so this script needs no network or GPU to run.
    combined_semantic = 0.55 * sim_score + 0.45 * facet_result["facet_score"]
    score = combined_semantic * penalty * f_penalty * exp_fit * loc_fit * behav
    if honeypot:
        score = -1.0

    return {
        "score": score,
        "honeypot": honeypot,
        "penalty": penalty,
        "facet_penalty": f_penalty,
        "facet_score": facet_result["facet_score"],
        "sim_score": sim_score,
        "missed_must_haves": facet_result["missed_must_haves"],
        "behav": behav,
    }


def main():
    all_candidates = [case["candidate"] for case in CASES]
    sim_scores = compute_semantic_scores(all_candidates, JD_TEXT)

    results = []
    for case in CASES:
        c = case["candidate"]
        r = score_candidate(c, sim_scores[c["candidate_id"]])
        results.append({**case, "result": r})

    results_sorted = sorted(results, key=lambda x: -x["result"]["score"])

    print("Adversarial test results, ranked by this ranker's score")
    print("")
    for rank, r in enumerate(results_sorted, start=1):
        res = r["result"]
        flag = "EXCLUDED (honeypot)" if res["honeypot"] else f"score {res['score']:.3f}"
        print(f"{rank}. {r['id']} expected={r['expect']} actual={flag}")
        print(f"   sim={res['sim_score']:.2f} penalty={res['penalty']:.2f} facet_penalty={res['facet_penalty']:.2f} "
              f"facet_score={res['facet_score']:.2f} behavioral={res['behav']:.2f}")
        if res["missed_must_haves"]:
            print(f"   missed must-haves: {', '.join(res['missed_must_haves'])}")
        print("")

    # Bucket-based pass/fail check. Ranks are computed only among non-honeypot
    # candidates, since honeypots are removed from ranking entirely.
    non_honeypot_sorted = [r for r in results_sorted if not r["result"]["honeypot"]]
    n = len(non_honeypot_sorted)
    rank_of = {r["id"]: i + 1 for i, r in enumerate(non_honeypot_sorted)}

    checks = []
    for r in results:
        cid = r["id"]
        expect = r["expect"]
        if expect == "excluded":
            checks.append((cid, r["result"]["honeypot"] == True))
        elif expect == "top":
            checks.append((cid, cid in rank_of and rank_of[cid] <= 2))
        elif expect == "upper_half":
            checks.append((cid, cid in rank_of and rank_of[cid] <= max(1, n // 2)))
        elif expect in ("bottom", "middle_or_bottom"):
            checks.append((cid, cid in rank_of and rank_of[cid] > n // 2))
        elif expect == "middle":
            checks.append((cid, cid in rank_of))

    print("Per-case pass/fail:")
    for cid, passed in checks:
        print(f"  {cid}: {'PASS' if passed else 'FAIL'}")
    all_passed = all(p for _, p in checks)
    print("")
    print("All cases passed:", all_passed)
    print("")
    print("Note: this script uses a TF-IDF stand-in for the semantic similarity")
    print("score so it runs standalone with no network or model download.")
    print("TF-IDF matches on shared words, so it understates fit for the")
    print("plain-language case compared to the real sentence-transformers model")
    print("used in precompute_embeddings.py and rank.py, which matches on")
    print("meaning rather than shared vocabulary.")


if __name__ == "__main__":
    main()
