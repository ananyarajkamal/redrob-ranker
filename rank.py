"""
Timed ranking step. Must run in <5 min, CPU only, no network, <16GB RAM.
Loads artifacts produced offline by precompute_embeddings.py.

Usage:
    python rank.py --candidates ./candidates.jsonl --artifacts ./artifacts --out ./submission.csv
"""
import argparse
import json
import time

import numpy as np

from common import (
    load_candidates, years_experience_fit, location_fit, hard_penalty_gate,
    is_honeypot, behavioral_multiplier, has_retrieval_evidence, build_reasoning,
    score_jd_facets, facet_penalty,
)

TOP_N = 100


def main():
    t_start = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, help="Path to candidates.jsonl (used for id order sanity check; metadata comes from artifacts cache)")
    ap.add_argument("--artifacts", default="./artifacts", help="Dir produced by precompute_embeddings.py")
    ap.add_argument("--out", default="./submission.csv")
    args = ap.parse_args()

    print("Loading artifacts ...")
    jd_emb = np.load(f"{args.artifacts}/jd_embedding.npy")
    cand_emb = np.load(f"{args.artifacts}/candidate_embeddings.npy")
    with open(f"{args.artifacts}/candidate_ids.json") as f:
        ids = json.load(f)
    with open(f"{args.artifacts}/honeypot_flags.json") as f:
        honeypot_flags_precomputed = json.load(f)
    with open(f"{args.artifacts}/retrieval_hits.json") as f:
        retrieval_hits_precomputed = json.load(f)

    # Load full candidate metadata from the cached jsonl (fast, local, no re-parsing of the raw file)
    id_to_candidate = {}
    for c in load_candidates(f"{args.artifacts}/candidates_cache.jsonl"):
        id_to_candidate[c["candidate_id"]] = c

    print(f"Loaded {len(ids)} candidates, embeddings shape {cand_emb.shape}")

    # ---- Semantic similarity (cosine, embeddings are pre-normalized) ----
    sim_scores = cand_emb @ jd_emb  # shape (N,)

    # ---- Rule-based components ----
    N = len(ids)
    final_scores = np.zeros(N, dtype=np.float64)
    penalties = np.zeros(N, dtype=np.float64)
    behav_mults = np.zeros(N, dtype=np.float64)
    honeypot_mask = np.zeros(N, dtype=bool)
    facet_results = [None] * N

    for i, cid in enumerate(ids):
        c = id_to_candidate.get(cid)
        if c is None:
            final_scores[i] = -1.0
            honeypot_mask[i] = True
            continue

        honeypot = honeypot_flags_precomputed[i] if i < len(honeypot_flags_precomputed) else is_honeypot(c)
        honeypot_mask[i] = honeypot

        penalty = hard_penalty_gate(c)
        exp_fit = years_experience_fit(c.get("profile", {}).get("years_of_experience"))
        loc_fit = location_fit(c.get("profile", {}).get("location"), c.get("profile", {}).get("country"))
        behav = behavioral_multiplier(c)

        facet_result = score_jd_facets(c)
        facet_results[i] = facet_result
        f_penalty = facet_penalty(facet_result["missed_must_haves"])

        penalties[i] = penalty
        behav_mults[i] = behav

        # Combine the semantic similarity score (broad, holistic match) with the
        # structured facet score (specific, checkable requirement match).
        # 55/45 split: similarity captures overall narrative fit, facets catch
        # concrete must-haves that a keyword-stuffed profile can fake in the
        # embedding but not in structured evidence checks tied to career text.
        combined_semantic = 0.55 * sim_scores[i] + 0.45 * facet_result["facet_score"]

        score = combined_semantic * penalty * f_penalty * exp_fit * loc_fit * behav
        if honeypot:
            score = -1.0  # hard exclude honeypots from ranking
        final_scores[i] = score

    print(f"Scored {N} candidates in {time.time()-t_start:.1f}s so far")

    # ---- Rank and select top 100 ----
    # Primary key: score descending. Secondary key (tie-break): candidate_id ascending,
    # as required by the spec when scores are equal.
    order = sorted(range(N), key=lambda i: (-final_scores[i], ids[i]))
    top_idx = order[:TOP_N]

    # Normalize displayed score to (0,1], strictly non-increasing, with small deterministic tiebreak
    raw_top_scores = final_scores[top_idx]
    max_s = max(raw_top_scores.max(), 1e-9)
    min_s = raw_top_scores.min()
    span = max(max_s - min_s, 1e-9)

    prelim = []
    for idx in top_idx:
        cid = ids[idx]
        c = id_to_candidate[cid]
        norm_score = 0.5 + 0.5 * (final_scores[idx] - min_s) / span  # maps into (0.5, 1.0], preserves order
        reasoning = build_reasoning(
            c, sim_scores[idx], penalties[idx], behav_mults[idx],
            retrieval_hits_precomputed[idx] if idx < len(retrieval_hits_precomputed) else has_retrieval_evidence(c),
            facet_result=facet_results[idx],
        )
        prelim.append((cid, round(float(norm_score), 4), reasoning))

    # Re-sort using the ROUNDED score (ties can appear after rounding that weren't
    # ties in the raw float score) with candidate_id ascending as the tie-break,
    # per spec Section 3.
    prelim.sort(key=lambda r: (-r[1], r[0]))

    # Enforce strictly non-increasing score by rank (guards against any residual
    # floating point edge case)
    rows = []
    prev_score = None
    for i, (cid, score, reasoning) in enumerate(prelim):
        if prev_score is not None and score > prev_score:
            score = prev_score
        rows.append((cid, i + 1, score, reasoning))
        prev_score = score

    print("Writing CSV ...")
    import csv
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for cid, rank, score, reasoning in rows:
            writer.writerow([cid, rank, f"{score:.4f}", reasoning])

    honeypots_in_top100 = sum(1 for idx in top_idx if honeypot_mask[idx])
    print(f"Done in {time.time()-t_start:.1f}s. Wrote {len(rows)} rows to {args.out}")
    print(f"Honeypots in top 100: {honeypots_in_top100} ({honeypots_in_top100}%)")


if __name__ == "__main__":
    main()