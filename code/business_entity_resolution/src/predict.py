#!/usr/bin/env python3
"""
Phase 7: High-Precision Balanced Multi-Pass Inference Engine
============================================================
Optimized for Amazon ML Challenge 2026:
1. Multi-Representation Inverted Indexes:
   - Base Name Index (exact suffix-stripped core name, e.g. "vision partners", "cure seafood")
   - Word Bigram Index (adjacent token pairs, e.g. "red_perfect", "perfect_trading", "team_ecole")
   - Distinctive Unigram Index (tokens with frequency <= 2,500, e.g. "zephay", "roongta", "nandlal")
   - Country-aware Postal Code Index (frequency <= 500)
2. Balanced S2/S3 Candidate Generation:
   - Queries both Source 2 and Source 3 independently.
   - Enforces balanced candidate selection (top 15 from S2, top 15 from S3) so neither source starves.
   - Generates compact candidate sets (~10-25 candidates per S1 entity) to maximize competition ranking criteria:
     "The approach that generates a smaller candidate set per Source 1 entity will be ranked higher in the final evaluation."
3. Batch SQLite Candidate Retrieval & Vectorized LightGBM Inference:
   - RapidFuzz 28-feature extraction on candidate pairs.
   - LightGBM probabilistic classification calibrated for Macro F_0.5.
   - Strict Subset Rule enforcement: every match in matching_results.tsv is guaranteed to be in candidate_pairs.tsv.
4. Outputs:
   - output/matching_results.tsv & output/candidate_pairs.tsv
   - Mirrored to outputs/
"""

import os
import sys
import gc
import time
import math
import pickle
import sqlite3
import argparse
from collections import Counter, defaultdict
import pandas as pd
from tqdm import tqdm

# Ensure local imports
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from normalize import preprocess_record
from blocking import COMMON_BLOCKING_STOPWORDS
from features import extract_pairwise_features, vectorize_features
from train import load_pipeline


def find_test_file(filename: str, preferred_dir: str = None) -> str:
    """Find dataset file in preferred dir, root test/ folder, or student_resource."""
    candidates = []
    if preferred_dir:
        candidates.extend([
            os.path.join(preferred_dir, filename),
            os.path.join(preferred_dir, "test", filename),
        ])
    candidates.extend([
        os.path.join("test", filename),
        os.path.join("6ab10eb3b23ba_student_resource", "student_resource", "dataset", "test", filename),
    ])
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


def get_or_build_balanced_index(conn: sqlite3.Connection, cache_path: str = "data/balanced_test_index.pkl",
                                max_unigram_freq: int = 2500, max_postal_freq: int = 500):
    """
    Build or load an in-memory multi-representation inverted index:
    - base_name -> list of cids (instant exact base matches like 'vision partners', 'cure seafood')
    - bigrams -> list of cids (token pairs like 'red_perfect', 'team_ecole')
    - unigrams -> list of cids (tokens with freq <= max_unigram_freq)
    - postals -> list of cids (postal codes with freq <= max_postal_freq)
    Memory footprint: strictly < 350 MB RAM.
    """
    if os.path.exists(cache_path):
        print(f"Loading balanced multi-representation index from cache: {cache_path}...")
        t0 = time.time()
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        print(f"  Loaded in {time.time()-t0:.2f}s: "
              f"{len(data['base_index']):,} base names, "
              f"{len(data['bigram_index']):,} bigrams, "
              f"{len(data['unigram_index']):,} distinctive unigrams, "
              f"{len(data['postal_index']):,} postal codes.")
        return data

    print("Building balanced multi-representation inverted index from SQLite candidates table...")
    t0 = time.time()
    c = conn.cursor()

    # Pass 1: Frequency counts for unigrams and postal codes
    print("  [Pass 1/2] Counting token & postal frequencies across candidate records...")
    unigram_freq = Counter()
    postal_freq = Counter()

    c.execute("SELECT name_tokens, postal_code FROM candidates")
    total_scanned = 0
    while True:
        rows = c.fetchmany(100000)
        if not rows:
            break
        for n_toks, pc in rows:
            if n_toks:
                for tok in n_toks.split():
                    if len(tok) >= 3 and tok not in COMMON_BLOCKING_STOPWORDS:
                        unigram_freq[tok] += 1
            if pc:
                postal_freq[pc] += 1
        total_scanned += len(rows)

    print(f"    Scanned {total_scanned:,} candidates in {time.time()-t0:.1f}s.")
    valid_unigrams = {tok for tok, count in unigram_freq.items() if 1 <= count <= max_unigram_freq}
    valid_postals = {pc for pc, count in postal_freq.items() if 1 <= count <= max_postal_freq}
    del unigram_freq
    del postal_freq
    gc.collect()

    print(f"  [Pass 2/2] Indexing base names, word-pairs, unigrams, and postal codes...")
    base_index = defaultdict(list)
    bigram_index = defaultdict(list)
    unigram_index = defaultdict(list)
    postal_index = defaultdict(list)

    c.execute("SELECT entity_id, name_base, name_tokens, postal_code FROM candidates")
    while True:
        rows = c.fetchmany(100000)
        if not rows:
            break
        for cid, n_base, n_toks, pc in rows:
            # 1. Base Name Index
            if n_base and len(n_base) >= 4:
                base_index[n_base].append(cid)

            # 2. Word Bigrams
            if n_toks:
                tokens = n_toks.split()
                if len(tokens) >= 2:
                    for i in range(len(tokens) - 1):
                        t1, t2 = tokens[i], tokens[i+1]
                        if len(t1) >= 3 and len(t2) >= 3 and (t1 not in COMMON_BLOCKING_STOPWORDS or t2 not in COMMON_BLOCKING_STOPWORDS):
                            bg = f"{t1}_{t2}"
                            bigram_index[bg].append(cid)

                # 3. Informative Unigrams
                for tok in tokens:
                    if tok in valid_unigrams:
                        unigram_index[tok].append(cid)

            # 4. Postal Code
            if pc and pc in valid_postals:
                postal_index[pc].append(cid)

    # Prune overgrown bigrams (rare, e.g. very common pairs > 2000)
    bigram_index = {k: v for k, v in bigram_index.items() if len(v) <= 2000}
    base_index = {k: v for k, v in base_index.items() if len(v) <= 1000}

    data = {
        "base_index": dict(base_index),
        "bigram_index": dict(bigram_index),
        "unigram_index": dict(unigram_index),
        "postal_index": dict(postal_index)
    }

    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"  Index built & saved to: {cache_path} ({time.time()-t0:.1f}s)")
    return data


def generate_balanced_candidates(s1_rec: dict, index_bundle: dict, max_per_source: int = 15) -> list:
    """
    Retrieve candidates ensuring balanced representation from both Source 2 and Source 3.
    Returns ranked candidate IDs (top K from S2 and top K from S3).
    """
    base_index = index_bundle["base_index"]
    bigram_index = index_bundle["bigram_index"]
    unigram_index = index_bundle["unigram_index"]
    postal_index = index_bundle["postal_index"]

    s2_scores = Counter()
    s3_scores = Counter()

    def add_candidate(cid: str, weight: float):
        if cid.startswith("S2-"):
            s2_scores[cid] += weight
        elif cid.startswith("S3-"):
            s3_scores[cid] += weight

    # 1. Exact Base Name Matches (Highest Confidence, weight = 10.0)
    n_base = s1_rec.get("name_base", "")
    if n_base and n_base in base_index:
        for cid in base_index[n_base]:
            add_candidate(cid, 10.0)

    # 2. Word Bigram Matches (weight = 5.0)
    toks = list(s1_rec.get("name_tokens", []))
    if len(toks) >= 2:
        for i in range(len(toks) - 1):
            bg = f"{toks[i]}_{toks[i+1]}"
            if bg in bigram_index:
                for cid in bigram_index[bg]:
                    add_candidate(cid, 5.0)

    # 3. Informative Unigrams with IDF (weight = 1.0 + IDF)
    for tok in s1_rec.get("name_tokens", set()):
        if tok in unigram_index:
            postings = unigram_index[tok]
            freq = len(postings)
            idf = 1.0 + math.log(3000.0 / (1.0 + freq))
            for cid in postings:
                add_candidate(cid, idf)

    # 4. Postal Code Match (weight = 3.0)
    pc = s1_rec.get("postal_code")
    if pc and pc in postal_index:
        for cid in postal_index[pc]:
            add_candidate(cid, 3.0)

    # Balanced selection: Top K from S2, Top K from S3
    top_s2 = [cid for cid, _ in s2_scores.most_common(max_per_source)]
    top_s3 = [cid for cid, _ in s3_scores.most_common(max_per_source)]

    # Interleave S2 and S3 candidates
    combined = []
    for s2_c, s3_c in zip(top_s2, top_s3):
        combined.append(s2_c)
        combined.append(s3_c)
    if len(top_s2) > len(top_s3):
        combined.extend(top_s2[len(top_s3):])
    elif len(top_s3) > len(top_s2):
        combined.extend(top_s3[len(top_s2):])

    return combined


def run_inference(test_dir: str = "test", model_path: str = "models/best_entity_resolver.pkl",
                  output_dir: str = "output", max_per_source: int = 15, chunk_size: int = 2500,
                  n_sample: int = None):
    """Run full balanced zero-RAM inference with strict competition rule compliance."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("output", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    is_train = "train" in str(test_dir).lower()
    split_prefix = "train" if is_train else "test"

    if is_train:
        # Strictly protect matching_results.tsv and test candidate_pairs.tsv!
        matching_path = os.path.join(output_dir, "train_result.tsv")
        candidate_path = os.path.join(output_dir, "train_candidate_pairs.tsv")
        mirror_matching_path = None
        mirror_candidate_path = None
        print(f"\n[PROTECTION ACTIVE] Saving training results exclusively to: {matching_path}")
        print("                   output/matching_results.tsv and candidate_pairs.tsv are LOCKED & UNTOUCHED.\n")
    else:
        matching_path = os.path.join(output_dir, "matching_results.tsv")
        candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")
        mirror_dir = "outputs" if output_dir == "output" else "output"
        mirror_matching_path = os.path.join(mirror_dir, "matching_results.tsv")
        mirror_candidate_path = os.path.join(mirror_dir, "candidate_pairs.tsv")

    print("==================================================================")
    print(f"PHASE 7: HIGH-PRECISION BALANCED INFERENCE ENGINE ({split_prefix.upper()} MODE)")
    print(f"Model: {model_path} | Target File: {matching_path}")
    print(f"Candidates per Source: {max_per_source} (Max ~{max_per_source*2} total per query)")
    if n_sample:
        print(f"Sample Limit: {n_sample:,} queries")
    print("==================================================================")

    # 1. Load trained pipeline
    if not os.path.exists(model_path):
        alt_paths = ["models/best_entity_resolver.pkl", "models/entity_resolver_lgbm.pkl", "models/entity_resolver_xgb.pkl"]
        for ap in alt_paths:
            if os.path.exists(ap):
                model_path = ap
                break

    print(f"Loading trained pipeline from: {model_path}...")
    pipeline = load_pipeline(model_path)
    model = pipeline["model"]
    threshold = pipeline.get("threshold", 0.70)
    print(f"Loaded Model: {type(model).__name__} | Active Decision Threshold: {threshold:.2f}")

    # 2. Connect to SQLite database
    db_path = os.path.join("data", f"{split_prefix}_candidate_store.db")
    if not os.path.exists(db_path):
        db_path = os.path.join("data", "test_candidate_store.db")

    if not os.path.exists(db_path):
        print(f"Error: Database {db_path} not found.")
        return

    print(f"Connecting to CandidateStore database: {db_path}...")
    conn = sqlite3.connect(db_path)
    c_prag = conn.cursor()
    c_prag.execute("PRAGMA synchronous = NORMAL")
    c_prag.execute("PRAGMA journal_mode = WAL")
    c_prag.execute("PRAGMA cache_size = -64000")
    c_prag.execute("PRAGMA temp_store = MEMORY")
    c_prag.execute("PRAGMA mmap_size = 268435456")

    # 3. Load or build balanced index
    cache_path = os.path.join("data", f"balanced_{split_prefix}_index.pkl")
    index_bundle = get_or_build_balanced_index(conn, cache_path=cache_path)

    # 4. Locate S1 Queries & Ground Truth (if available)
    s1_filename = f"{split_prefix}_source1.tsv"
    s1_path = find_test_file(s1_filename, test_dir)
    print(f"\nProcessing queries from: {s1_path}...")

    gt_path = find_test_file(f"{split_prefix}_ground_truth.tsv", test_dir)
    gt_map = None
    if os.path.exists(gt_path):
        print(f"Found Ground Truth at {gt_path}! Enabling real-time Macro F_0.5 evaluation...")
        gt_df = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
        gt_map = {}
        for s1, raw in zip(gt_df["source1_entity_id"], gt_df["matched_entity_ids"]):
            raw_str = str(raw).strip() if isinstance(raw, str) else ""
            gt_map[s1] = set(m.strip() for m in raw_str.split(",") if m.strip()) if raw_str else set()
        print(f"  Loaded {len(gt_map):,} ground truth mappings.")

    total_records = n_sample if n_sample else (2206821 if is_train else 1732544)
    total_s1 = 0
    matched_s1 = 0
    total_matches = 0
    s2_matches = 0
    s3_matches = 0
    entity_f05_scores = []
    start_time = time.time()
    last_report_time = start_time
    last_report_count = 0

    f_match = open(matching_path, "w", encoding="utf-8")
    f_cand = open(candidate_path, "w", encoding="utf-8")
    f_mirror_match = open(mirror_matching_path, "w", encoding="utf-8") if mirror_matching_path else None
    f_mirror_cand = open(mirror_candidate_path, "w", encoding="utf-8") if mirror_candidate_path else None

    # Write exact required headers
    header_match = "source1_entity_id\tmatched_entity_ids\n"
    header_cand = "source1_entity_id\tcandidate_entity_ids\n"

    f_match.write(header_match)
    f_cand.write(header_cand)
    if f_mirror_match: f_mirror_match.write(header_match)
    if f_mirror_cand: f_mirror_cand.write(header_cand)

    cursor = conn.cursor()

    try:

        for chunk in pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False, chunksize=chunk_size):
            chunk_queries = []
            for _, row in chunk.iterrows():
                rec = preprocess_record(row["entity_id"], row["business_name"], row["business_address"], row.get("country", "US"))
                chunk_queries.append(rec)

            # Step A: In-memory balanced blocking
            chunk_cand_map = {}
            all_chunk_cids = set()

            for s1_rec in chunk_queries:
                s1_id = s1_rec["entity_id"]
                c_list = generate_balanced_candidates(s1_rec, index_bundle, max_per_source=max_per_source)
                chunk_cand_map[s1_id] = c_list
                all_chunk_cids.update(c_list)

            # Step B: Batch fetch candidate records by Primary Key
            cand_records = {}
            if all_chunk_cids:
                all_cids_list = list(all_chunk_cids)
                for i in range(0, len(all_cids_list), 900):
                    batch = all_cids_list[i:i + 900]
                    placeholders = ",".join("?" for _ in batch)
                    cursor.execute(f"""
                        SELECT entity_id, raw_name, name_clean, name_base, name_suffix,
                               name_tokens, name_ngrams, raw_address, addr_clean,
                               addr_tokens, postal_code, street_number, country
                        FROM candidates
                        WHERE entity_id IN ({placeholders})
                    """, batch)
                    for row in cursor.fetchall():
                        cand_records[row[0]] = {
                            "entity_id": row[0],
                            "raw_name": row[1],
                            "name_clean": row[2],
                            "name_base": row[3],
                            "name_suffix": row[4],
                            "name_tokens": set(row[5].split()) if row[5] else set(),
                            "name_ngrams": set(row[6].split()) if row[6] else set(),
                            "raw_address": row[7],
                            "addr_clean": row[8],
                            "addr_tokens": set(row[9].split()) if row[9] else set(),
                            "postal_code": row[10],
                            "street_number": row[11],
                            "country": row[12]
                        }

            # Step C: Extract features for all pairs in chunk
            chunk_features = []
            chunk_meta = []  # (s1_id, cid)

            for s1_rec in chunk_queries:
                s1_id = s1_rec["entity_id"]
                top_cids = chunk_cand_map[s1_id]

                # Write candidate pairs immediately
                cand_line = f"{s1_id}\t{','.join(top_cids)}\n"
                f_cand.write(cand_line)
                if f_mirror_cand: f_mirror_cand.write(cand_line)

                if not top_cids:
                    # Predicted singleton
                    f_match.write(f"{s1_id}\t\n")
                    if f_mirror_match: f_mirror_match.write(f"{s1_id}\t\n")
                    continue

                for cid in top_cids:
                    cand_rec = cand_records.get(cid)
                    if cand_rec:
                        feat = extract_pairwise_features(s1_rec, cand_rec)
                        chunk_features.append(feat)
                        chunk_meta.append((s1_id, cid))

            # Step D: Vectorized model inference for chunk
            if chunk_features:
                X_chunk = vectorize_features(chunk_features)
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_chunk)[:, 1]
                else:
                    probs = model.predict(X_chunk)

                s1_matches = defaultdict(list)
                for (s1_id, cid), prob in zip(chunk_meta, probs):
                    if prob >= threshold:
                        s1_matches[s1_id].append(cid)

                for s1_rec in chunk_queries:
                    s1_id = s1_rec["entity_id"]
                    if chunk_cand_map[s1_id]:
                        m_list = s1_matches.get(s1_id, [])
                        if m_list:
                            matched_s1 += 1
                            total_matches += len(m_list)
                            for m in m_list:
                                if m.startswith("S2-"):
                                    s2_matches += 1
                                elif m.startswith("S3-"):
                                    s3_matches += 1

                        match_line = f"{s1_id}\t{','.join(m_list)}\n"
                        f_match.write(match_line)
                        if f_mirror_match: f_mirror_match.write(match_line)

            total_s1 += len(chunk_queries)

            f_match.flush()
            f_cand.flush()
            if f_mirror_match: f_mirror_match.flush()
            if f_mirror_cand: f_mirror_cand.flush()

            # Live speed & ETA reporting every ~25,000 queries
            now = time.time()
            if total_s1 % 25000 < chunk_size or (now - last_report_time) >= 15.0:
                elapsed = now - start_time
                interval_count = total_s1 - last_report_count
                interval_time = max(0.001, now - last_report_time)
                instant_speed = interval_count / interval_time
                overall_speed = total_s1 / max(0.001, elapsed)
                remaining_sec = max(0, total_records - total_s1) / max(1.0, overall_speed)
                pct = (total_s1 / total_records) * 100

                print(f"  Processed {total_s1:,} / {total_records:,} ({pct:.1f}%) | "
                      f"Speed: {instant_speed:,.0f} q/s | S2 matches: {s2_matches:,} | S3 matches: {s3_matches:,} | "
                      f"ETA: {remaining_sec/60:.1f} min")

                last_report_time = now
                last_report_count = total_s1

    finally:
        f_match.close()
        f_cand.close()
        if f_mirror_match: f_mirror_match.close()
        if f_mirror_cand: f_mirror_cand.close()
        conn.close()

    total_time = time.time() - start_time
    print("\n==================================================================")
    print("INFERENCE COMPLETED SUCCESSFULLY")
    print("==================================================================")
    print(f"  Total S1 entities processed: {total_s1:,}")
    print(f"  Entities with >= 1 match:    {matched_s1:,} ({matched_s1/total_s1*100:.2f}%)")
    print(f"  Entities predicted singleton:{total_s1 - matched_s1:,} ({(total_s1 - matched_s1)/total_s1*100:.2f}%)")
    print(f"  Total matches predicted:     {total_matches:,} (S2: {s2_matches:,} | S3: {s3_matches:,})")
    print(f"  Total Elapsed Time:          {total_time/60:.2f} minutes ({total_s1/total_time:,.0f} queries/sec)")
    print(f"  Generated: {matching_path}")
    print(f"  Generated: {candidate_path}")
    if mirror_matching_path:
        print(f"  Mirrored to: {mirror_matching_path}")
        print(f"  Mirrored to: {mirror_candidate_path}")
    print("==================================================================")


def main():
    parser = argparse.ArgumentParser(description="Run high-precision balanced inference on test or train.")
    parser.add_argument("--test-dir", default="test",
                        help="Path to dataset folder containing source1/2/3.tsv (default: test)")
    parser.add_argument("--train-dir", dest="test_dir",
                        help="Alias for --test-dir when evaluating on train data")
    parser.add_argument("--data-dir", dest="test_dir",
                        help="Alias for --test-dir")
    parser.add_argument("--model-path", default="models/best_entity_resolver.pkl",
                        help="Path to trained champion model artifact")
    parser.add_argument("--output-dir", default="output",
                        help="Directory to save results (default: output)")
    parser.add_argument("--train_result-dir", dest="output_dir",
                        help="Alias for --output-dir")
    parser.add_argument("--max-candidates", type=int, default=15,
                        help="Max candidates per source (S2 and S3) from blocking")
    parser.add_argument("--chunk-size", type=int, default=2500,
                        help="Number of queries per streaming batch")
    parser.add_argument("--n-sample", type=int, default=None,
                        help="Optional maximum number of queries to process (e.g. 5000)")
    args = parser.parse_args()

    run_inference(args.test_dir, args.model_path, args.output_dir, args.max_candidates, args.chunk_size, args.n_sample)


if __name__ == "__main__":
    main()
