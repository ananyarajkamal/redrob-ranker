"""
Offline precompute step. NOT part of the timed 5-minute ranking budget.
Run this once ahead of time; rank.py just loads the artifacts it produces.

Usage:
    python precompute_embeddings.py --candidates ./candidates.jsonl --out_dir ./artifacts
"""
import argparse
import json
import time

import numpy as np

from common import load_candidates, candidate_text, JD_TEXT, is_honeypot, has_retrieval_evidence

MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, local, no API key needed
BATCH_SIZE = 256


def embed_sbert(texts, jd_text):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    jd_emb = model.encode([jd_text], normalize_embeddings=True)[0]
    cand_emb = model.encode(
        texts, batch_size=BATCH_SIZE, normalize_embeddings=True,
        show_progress_bar=True, convert_to_numpy=True
    )
    return jd_emb, cand_emb.astype(np.float32)


def embed_tfidf(texts, jd_text, n_components=256):
    """
    Pure scikit-learn fallback: TF-IDF + truncated SVD (LSA).
    No network / no model download required. Use if sentence-transformers /
    HuggingFace Hub is unreachable (e.g. offline grading environment).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize

    vectorizer = TfidfVectorizer(max_features=50000, stop_words="english", ngram_range=(1, 2))
    all_texts = [jd_text] + texts
    tfidf = vectorizer.fit_transform(all_texts)

    n_comp = min(n_components, tfidf.shape[0] - 1, tfidf.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    reduced = svd.fit_transform(tfidf)
    reduced = normalize(reduced)

    jd_emb = reduced[0].astype(np.float32)
    cand_emb = reduced[1:].astype(np.float32)
    return jd_emb, cand_emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out_dir", default="./artifacts")
    ap.add_argument("--limit", type=int, default=None, help="For testing on a subset")
    ap.add_argument("--engine", choices=["sbert", "tfidf"], default="sbert",
                     help="sbert = sentence-transformers (needs network once to download model). "
                          "tfidf = pure scikit-learn fallback, no network needed.")
    args = ap.parse_args()

    import os
    os.makedirs(args.out_dir, exist_ok=True)

    ids = []
    texts = []
    honeypot_flags = []
    retrieval_hits = []
    metadata = []

    t0 = time.time()
    n = 0
    for c in load_candidates(args.candidates):
        ids.append(c["candidate_id"])
        texts.append(candidate_text(c))
        honeypot_flags.append(is_honeypot(c))
        retrieval_hits.append(has_retrieval_evidence(c))
        metadata.append(c)  # kept in memory; for 100K candidates this is fine (~1-2GB)
        n += 1
        if args.limit and n >= args.limit:
            break
        if n % 10000 == 0:
            print(f"  loaded {n} candidates ({time.time()-t0:.1f}s)")

    print(f"Loaded {n} candidates in {time.time()-t0:.1f}s. Embedding with engine={args.engine} ...")
    t1 = time.time()
    if args.engine == "sbert":
        jd_emb, embeddings = embed_sbert(texts, JD_TEXT)
    else:
        jd_emb, embeddings = embed_tfidf(texts, JD_TEXT)
    print(f"Embedded {n} candidates in {time.time()-t1:.1f}s")

    np.save(f"{args.out_dir}/jd_embedding.npy", jd_emb)
    np.save(f"{args.out_dir}/candidate_embeddings.npy", embeddings.astype(np.float32))
    with open(f"{args.out_dir}/engine.txt", "w") as f:
        f.write(args.engine)
    with open(f"{args.out_dir}/candidate_ids.json", "w") as f:
        json.dump(ids, f)
    with open(f"{args.out_dir}/honeypot_flags.json", "w") as f:
        json.dump(honeypot_flags, f)
    with open(f"{args.out_dir}/retrieval_hits.json", "w") as f:
        json.dump(retrieval_hits, f)
    # Save raw candidate JSON as jsonl too, so rank.py can reload metadata fast
    with open(f"{args.out_dir}/candidates_cache.jsonl", "w") as f:
        for c in metadata:
            f.write(json.dumps(c) + "\n")

    print(f"Done. Artifacts written to {args.out_dir}")


if __name__ == "__main__":
    main()
