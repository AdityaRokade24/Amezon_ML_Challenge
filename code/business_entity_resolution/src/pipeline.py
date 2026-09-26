#!/usr/bin/env python3
"""
Pipeline Orchestrator: End-to-End Experiment & Model Builder
=============================================================
Runs training, blocking evaluation, threshold tuning, and model export:
1. Loads validation split & candidate pools
2. Runs multi-pass blocking & measures candidate recall
3. Generates training pairs (ground truth positives + mined hard negatives)
4. Vectorizes features with RapidFuzz
5. Trains LightGBM / Logistic Regression
6. Optimizes threshold for exact Macro F_0.5
7. Saves the best model pipeline to models/
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

from normalize import preprocess_record
from blocking import MultiPassBlocker, evaluate_blocking_recall
from features import extract_pairwise_features, vectorize_features
from train import train_models, save_pipeline
from evaluate import compute_macro_f05, optimize_threshold


def run_experiment(data_dir: str, val_dir: str = "data/val", n_sample: int = 5000, model_type: str = "lightgbm", output_model: str = "models/entity_resolver_lgbm.pkl"):
    print("==================================================================")
    print("RUNNING END-TO-END PIPELINE EXPERIMENT")
    print(f"Model: {model_type} | Sample S1 Queries: {n_sample:,}")
    print("==================================================================")

    # 1. Load validation data
    val_s1_path = os.path.join(val_dir, "val_source1.tsv")
    val_gt_path = os.path.join(val_dir, "val_ground_truth.tsv")

    if not os.path.exists(val_s1_path) or not os.path.exists(val_gt_path):
        print("Validation files not found! Please run audit.py first.")
        return

    print("Loading validation records...")
    df_val_s1 = pd.read_csv(val_s1_path, sep="\t", dtype=str, nrows=n_sample)
    df_val_gt = pd.read_csv(val_gt_path, sep="\t", dtype=str, keep_default_na=False)
    
    # Ground truth mapping: s1_id -> set(matched_ids)
    gt_map = {}
    for _, row in df_val_gt.iterrows():
        s1 = row["source1_entity_id"]
        raw = row["matched_entity_ids"].strip()
        gt_map[s1] = set(m.strip() for m in raw.split(",") if m.strip()) if raw else set()

    # Preprocess S1 query records
    val_s1_records = []
    needed_positive_ids = set()
    for _, row in df_val_s1.iterrows():
        s1_rec = preprocess_record(row["entity_id"], row["business_name"], row["business_address"], row.get("country", "US"))
        val_s1_records.append(s1_rec)
        needed_positive_ids.update(gt_map.get(row["entity_id"], set()))

    print(f"  Loaded {len(val_s1_records):,} S1 queries with {len(needed_positive_ids):,} associated true positive match IDs.")

    # 2. Load candidate pool (Stream from train_source2 and train_source3 to find positives + negatives)
    cand_records = {}
    print("Streaming candidate records from train_source2 and train_source3...")
    for src_file in ["train_source2.tsv", "train_source3.tsv"]:
        p = os.path.join(data_dir, "train", src_file)
        if not os.path.exists(p):
            continue
        print(f"  Scanning {src_file}...")
        for chunk in pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, chunksize=250000):
            # Keep all records that are known true positives for our validation queries
            matches = chunk[chunk["entity_id"].isin(needed_positive_ids)]
            for _, row in matches.iterrows():
                cand_records[row["entity_id"]] = preprocess_record(
                    row["entity_id"], row["business_name"], row["business_address"], row.get("country", "US")
                )
            # Also keep a background pool of candidates for realistic negative mining
            if len(cand_records) < len(needed_positive_ids) + 50000:
                sample_chunk = chunk.sample(n=min(len(chunk), 5000), random_state=42)
                for _, row in sample_chunk.iterrows():
                    if row["entity_id"] not in cand_records:
                        cand_records[row["entity_id"]] = preprocess_record(
                            row["entity_id"], row["business_name"], row["business_address"], row.get("country", "US")
                        )

    print(f"  Candidate pool indexed: {len(cand_records):,} records (positives captured: {len(needed_positive_ids.intersection(cand_records.keys())):,}/{len(needed_positive_ids):,})")

    # 3. Fit Blocker
    blocker = MultiPassBlocker(max_token_freq=5000, min_token_len=3)
    blocker.fit_candidates(list(cand_records.values()))

    # Evaluate blocking recall
    evaluate_blocking_recall(val_s1_records, gt_map, blocker, max_k=40)

    # 4. Generate Training Pairs & Hard Negatives
    print("\nExtracting feature vectors for candidate pairs...")
    X_list = []
    y_list = []
    val_candidate_scores = []

    for s1_rec in tqdm(val_s1_records, desc="Extracting features"):
        s1_id = s1_rec["entity_id"]
        true_matches = gt_map.get(s1_id, set())

        # Retrieve blocking candidates
        cand_ids = blocker.generate_candidates(s1_rec, max_candidates_per_query=40)

        # Include true positives if any were missed by blocking (for supervised training)
        all_eval_cands = set(cand_ids)
        for tp_id in true_matches:
            if tp_id in cand_records:
                all_eval_cands.add(tp_id)

        for cid in all_eval_cands:
            c_rec = cand_records.get(cid)
            if not c_rec:
                continue
            is_match = 1.0 if cid in true_matches else 0.0
            feat_dict = extract_pairwise_features(s1_rec, c_rec)
            X_list.append(feat_dict)
            y_list.append(is_match)

    X_mat = vectorize_features(X_list)
    y_arr = np.array(y_list, dtype=np.float32)
    print(f"  Feature matrix constructed: {X_mat.shape} (Positives: {y_arr.sum():,}, Negatives: {len(y_arr) - y_arr.sum():,})")

    # Split into train and test folds
    n_pairs = len(X_mat)
    perm = np.random.RandomState(42).permutation(n_pairs)
    split_idx = int(0.70 * n_pairs)
    train_idx, val_idx = perm[:split_idx], perm[split_idx:]

    X_train, y_train = X_mat[train_idx], y_arr[train_idx]
    X_val, y_val = X_mat[val_idx], y_arr[val_idx]

    # 5. Train Model
    model = train_models(X_train, y_train, model_type=model_type)

    # 6. Score validation set & optimize threshold
    val_probs = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_val)

    # Collect pairs for threshold optimization
    val_pairs = []
    cur_idx = 0
    for s1_rec in val_s1_records:
        s1_id = s1_rec["entity_id"]
        true_matches = gt_map.get(s1_id, set())
        cand_ids = blocker.generate_candidates(s1_rec, max_candidates_per_query=40)
        all_eval_cands = set(cand_ids)
        for tp_id in true_matches:
            if tp_id in cand_records:
                all_eval_cands.add(tp_id)
        for cid in all_eval_cands:
            if cid in cand_records:
                if cur_idx < len(X_mat):
                    # Compute probability
                    feat = extract_pairwise_features(s1_rec, cand_records[cid])
                    vec = vectorize_features([feat])
                    prob = float(model.predict_proba(vec)[0, 1])
                    val_pairs.append((s1_id, cid, prob))
                cur_idx += 1

    # Optimize threshold on Macro F_0.5
    sub_gt = {rec["entity_id"]: gt_map.get(rec["entity_id"], set()) for rec in val_s1_records}
    best_thresh, best_f05, best_details = optimize_threshold(val_pairs, sub_gt)

    # 7. Save pipeline artifact
    save_pipeline(model, best_thresh, output_model)
    print(f"\nExperiment finished with Macro F_0.5: {best_f05:.4f} at threshold {best_thresh:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Run pipeline experiment.")
    parser.add_argument("--data-dir", default="6ab10eb3b23ba_student_resource/student_resource/dataset",
                        help="Path to dataset root")
    parser.add_argument("--val-dir", default="data/val", help="Path to validation split")
    parser.add_argument("--n-sample", type=int, default=2000, help="Number of S1 queries to evaluate")
    parser.add_argument("--model-type", default="lightgbm", choices=["lightgbm", "logistic_regression", "random_forest", "xgboost"],
                        help="Model architecture to train")
    parser.add_argument("--output-model", default="models/entity_resolver_lgbm.pkl", help="Where to save model artifact")
    args = parser.parse_args()

    run_experiment(args.data_dir, args.val_dir, n_sample=args.n_sample, model_type=args.model_type, output_model=args.output_model)


if __name__ == "__main__":
    main()
