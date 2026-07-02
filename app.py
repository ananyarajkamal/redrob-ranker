"""
FastAPI web interface for the Ranker.

Speed strategy:
  - If precomputed artifacts exist -> use them (17s, same as rank.py)
  - If not -> fall back to TF-IDF on the uploaded file (slower)

Run:
    python -m uvicorn app:app --port 8000
"""

import csv
import io
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

app = FastAPI(title="Ranker")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    return (BASE_DIR / "static" / "index.html").read_text(encoding="utf-8")


@app.post("/api/rank")
async def rank(file: UploadFile = File(...), top_n: int = Form(100)):
    if not file.filename.endswith(".jsonl"):
        raise HTTPException(status_code=400, detail="Upload a .jsonl file.")

    top_n = max(1, min(top_n, 10000))
    content = await file.read()
    if not content.strip():
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="wb") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        from common import (
            JD_TEXT, behavioral_multiplier, build_reasoning,
            candidate_text, facet_penalty, hard_penalty_gate,
            has_retrieval_evidence, is_honeypot, load_candidates,
            location_fit, score_jd_facets, years_experience_fit,
        )

        candidates = list(load_candidates(tmp_path))
        if not candidates:
            raise HTTPException(status_code=400, detail="No valid candidate records found.")

        artifacts_dir = BASE_DIR / "artifacts"
        cache_file    = artifacts_dir / "candidates_cache.jsonl"
        jd_emb_file   = artifacts_dir / "jd_embedding.npy"
        cand_emb_file = artifacts_dir / "candidate_embeddings.npy"
        ids_file      = artifacts_dir / "candidate_ids.json"
        hp_file       = artifacts_dir / "honeypot_flags.json"
        rh_file       = artifacts_dir / "retrieval_hits.json"

        use_artifacts = all(
            p.exists() for p in [jd_emb_file, cand_emb_file, ids_file, hp_file, rh_file, cache_file]
        )

        if use_artifacts:
            # Fast path: use precomputed sentence-transformer embeddings
            jd_emb   = np.load(str(jd_emb_file))
            cand_emb = np.load(str(cand_emb_file))
            with open(ids_file) as f:   artifact_ids   = json.load(f)
            with open(hp_file) as f:    honeypot_flags = json.load(f)
            with open(rh_file) as f:    retrieval_hits = json.load(f)

            # Build lookup from cache (fast local read, not the raw upload)
            id_to_cand = {}
            for c in load_candidates(str(cache_file)):
                id_to_cand[c["candidate_id"]] = c

            # Filter to only the candidates present in the uploaded file
            uploaded_ids = {c.get("candidate_id") for c in candidates}
            id_to_idx = {cid: i for i, cid in enumerate(artifact_ids)}

            sims = cand_emb @ jd_emb
        else:
            # Slow path: TF-IDF + SVD (no artifacts)
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.preprocessing import normalize

            texts  = [candidate_text(c) for c in candidates]
            vec    = TfidfVectorizer(max_features=5000, sublinear_tf=True, ngram_range=(1, 2))
            matrix = vec.fit_transform([JD_TEXT] + texts)
            k      = min(100, matrix.shape[1] - 1, matrix.shape[0] - 1)
            svd    = TruncatedSVD(n_components=k, random_state=42)
            red    = normalize(svd.fit_transform(matrix), norm="l2")
            sims   = red[1:] @ red[0]

        scored = []
        honeypots_excluded = 0

        for i, c in enumerate(candidates):
            cid = c.get("candidate_id", f"CAND_{i:07d}")

            if use_artifacts:
                idx = id_to_idx.get(cid)
                if idx is None:
                    continue
                sim = float(sims[idx])
                hp_flag = honeypot_flags[idx] if idx < len(honeypot_flags) else is_honeypot(c)
                rh = retrieval_hits[idx] if idx < len(retrieval_hits) else has_retrieval_evidence(c)
                # Use cached candidate if available (has full data)
                c = id_to_cand.get(cid, c)
            else:
                sim     = float(sims[i])
                hp_flag = is_honeypot(c)
                rh      = has_retrieval_evidence(c)

            if hp_flag:
                honeypots_excluded += 1
                continue

            profile      = c.get("profile", {})
            facet_result = score_jd_facets(c)
            fp           = facet_penalty(facet_result["missed_must_haves"])
            combined_sem = 0.55 * sim + 0.45 * facet_result["facet_score"]
            hp           = hard_penalty_gate(c)
            exp_fit      = years_experience_fit(profile.get("years_of_experience"))
            loc_fit      = location_fit(profile.get("location", ""), profile.get("country", ""))
            behav        = behavioral_multiplier(c)
            raw_score    = combined_sem * hp * fp * exp_fit * loc_fit * behav

            reasoning = build_reasoning(c, sim, hp, behav, rh, facet_result)

            scored.append({
                "candidate_id": cid,
                "name":         profile.get("name", ""),
                "title":        profile.get("current_title", ""),
                "location":     profile.get("location", ""),
                "years_exp":    profile.get("years_of_experience"),
                "raw_score":    raw_score,
                "reasoning":    reasoning,
                "facet_hits":   facet_result["facet_hits"],
                "missed_must_haves": facet_result["missed_must_haves"],
                "sim":          sim,
                "hp":           hp,
                "behav":        behav,
            })

        if not scored:
            raise HTTPException(status_code=400, detail="No candidates passed the filters.")

        scored.sort(key=lambda x: (-x["raw_score"], x["candidate_id"]))
        top = scored[:top_n]

        raw_vals = [r["raw_score"] for r in top]
        span     = max(max(raw_vals) - min(raw_vals), 1e-9)
        min_s    = min(raw_vals)

        rows = []
        prev_score = None
        for rank_num, r in enumerate(top, 1):
            norm_score = round(0.5 + 0.5 * (r["raw_score"] - min_s) / span, 4)
            if prev_score is not None and norm_score > prev_score:
                norm_score = prev_score
            prev_score = norm_score
            rows.append({
                "candidate_id":      r["candidate_id"],
                "rank":              rank_num,
                "score":             norm_score,
                "reasoning":         r["reasoning"],
                "name":              r["name"],
                "title":             r["title"],
                "location":          r["location"],
                "years_exp":         r["years_exp"],
                "facet_hits":        r["facet_hits"],
                "missed_must_haves": r["missed_must_haves"],
            })

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in rows:
            writer.writerow([r["candidate_id"], r["rank"], f"{r['score']:.4f}", r["reasoning"]])

        return {
            "rows":               rows,
            "count":              len(rows),
            "total_input":        len(candidates),
            "honeypots_excluded": honeypots_excluded,
            "used_artifacts":     use_artifacts,
            "csv":                buf.getvalue(),
        }

    finally:
        os.unlink(tmp_path)
