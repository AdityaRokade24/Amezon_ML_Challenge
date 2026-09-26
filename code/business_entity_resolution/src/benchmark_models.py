#!/usr/bin/env python3
"""
Model Selection Engine: Benchmark All 4 Models on Exact Macro F_0.5
===================================================================
1. Trains 4 models on the EXACT same training data and features:
   - Logistic Regression
   - Random Forest
   - XGBoost
   - LightGBM
2. Independently optimizes the decision threshold (t*) for each model on Macro F_0.5.
3. Produces a comparative evaluation table.
4. Selects the champion model with the highest validation F_0.5 and saves it for test inference.
"""

import os
import sys

# Ensure local src modules can be imported
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from normalize import preprocess_record
from blocking import MultiPassBlocker
from features import extract_pairwise_features, vectorize_features
from train import train_models, save_pipeline
from evaluate import compute_macro_f05, optimize_threshold


def find_dataset_file(filename: str, preferred_dir: str = None) -> str:
    """Resolve dataset path across root train/ folder, custom dir, and student_resource."""
    candidates = []
    if preferred_dir:
        candidates.extend([
            os.path.join(preferred_dir, filename),
            os.path.join(preferred_dir, "train", filename),
        ])
    candidates.extend([
        os.path.join("train", filename),
        os.path.join("data", "val", filename),
        os.path.join("6ab10eb3b23ba_student_resource", "student_resource", "dataset", "train", filename),
    ])
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


def benchmark_all_models(data_dir: str = "train", val_dir: str = "data/val", n_sample: int = 2000, output_dir: str = "models"):
    print("==================================================================")
    print("PHASE 5: 4-MODEL BENCHMARK & SELECTION ON MACRO F_0.5")
    print(f"Sample S1 Queries: {n_sample:,}")
    print("Models to train: Logistic Regression, Random Forest, XGBoost, LightGBM")
    print("==================================================================")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Locate S1 queries and ground truth
    s1_path = find_dataset_file("train_source1.tsv", data_dir)
    gt_path = find_dataset_file("train_ground_truth.tsv", data_dir)

    print(f"\n[Step 1/5] Loading records from: {s1_path}")
    print(f"           Ground truth from:    {gt_path}")
    df_val_s1 = pd.read_csv(s1_path, sep="\t", dtype=str, nrows=n_sample)
    df_val_gt = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)

    gt_map = {}
    for _, row in df_val_gt.iterrows():
        s1 = row["source1_entity_id"]
        raw = row["matched_entity_ids"].strip()
        gt_map[s1] = set(m.strip() for m in raw.split(",") if m.strip()) if raw else set()

    val_s1_records = []
    needed_positive_ids = set()
    for _, row in df_val_s1.iterrows():
        s1_rec = preprocess_record(row["entity_id"], row["business_name"], row["business_address"], row.get("country", "US"))
        val_s1_records.append(s1_rec)
        needed_positive_ids.update(gt_map.get(row["entity_id"], set()))

    print(f"  Loaded {len(val_s1_records):,} S1 queries with {len(needed_positive_ids):,} associated true positive match IDs.")

    # 2. Stream candidate pool (Source 2 and Source 3)
    print("\n[Step 2/5] Indexing candidate records from train_source2 and train_source3...")
    cand_records = {}
    for src_file in ["train_source2.tsv", "train_source3.tsv"]:
        p = find_dataset_file(src_file, data_dir)
        if not os.path.exists(p):
            print(f"  Warning: {p} not found!")
            continue
        print(f"  Scanning {p}...")
        for chunk in pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, chunksize=250000):
            matches = chunk[chunk["entity_id"].isin(needed_positive_ids)]
            for _, row in matches.iterrows():
                cand_records[row["entity_id"]] = preprocess_record(
                    row["entity_id"], row["business_name"], row["business_address"], row.get("country", "US")
                )
            if len(cand_records) < len(needed_positive_ids) + 50000:
                sample_chunk = chunk.sample(n=min(len(chunk), 5000), random_state=42)
                for _, row in sample_chunk.iterrows():
                    if row["entity_id"] not in cand_records:
                        cand_records[row["entity_id"]] = preprocess_record(
                            row["entity_id"], row["business_name"], row["business_address"], row.get("country", "US")
                        )

    print(f"  Candidate pool ready: {len(cand_records):,} records.")

    # 3. Fit Blocker & Extract Features Once for All Models
    print("\n[Step 3/5] Building multi-pass blocking index and extracting pairwise feature vectors...")
    blocker = MultiPassBlocker(max_token_freq=6000, min_token_len=3)
    blocker.fit_candidates(list(cand_records.values()))

    # Leakage-Safe Entity Group Split (75% Train Entities, 25% Validation Entities)
    n_entities = len(val_s1_records)
    perm_ent = np.random.RandomState(42).permutation(n_entities)
    split_idx = int(0.75 * n_entities)
    train_eids = set(val_s1_records[i]["entity_id"] for i in perm_ent[:split_idx])
    val_eids = set(val_s1_records[i]["entity_id"] for i in perm_ent[split_idx:])

    train_X, train_y = [], []
    val_X, val_y = [], []
    val_pairs_meta = []

    for s1_rec in tqdm(val_s1_records, desc="Extracting features"):
        s1_id = s1_rec["entity_id"]
        is_train = s1_id in train_eids
        true_matches = gt_map.get(s1_id, set())

        cand_ids = blocker.generate_candidates(s1_rec, max_candidates_per_query=40)
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

            if is_train:
                train_X.append(feat_dict)
                train_y.append(is_match)
            else:
                val_X.append(feat_dict)
                val_y.append(is_match)
                val_pairs_meta.append((s1_id, cid))

    X_train = vectorize_features(train_X)
    y_train = np.array(train_y, dtype=np.float32)

    X_val = vectorize_features(val_X)
    y_val = np.array(val_y, dtype=np.float32)

    print(f"  Train set: {len(train_eids):,} entities -> {len(X_train):,} pairs (Positives: {y_train.sum():,})")
    print(f"  Validation set: {len(val_eids):,} entities -> {len(X_val):,} pairs (Positives: {y_val.sum():,})")

    sub_gt = {eid: gt_map.get(eid, set()) for eid in val_eids}

    # 4. Train & Evaluate All 4 Models
    print("\n[Step 4/5] Training and evaluating all 4 models on intact entity groups...")
    models_to_test = [
        ("Logistic Regression", "logistic_regression"),
        ("Random Forest", "random_forest"),
        ("XGBoost", "xgboost"),
        ("LightGBM", "lightgbm"),
    ]

    benchmark_results = []
    trained_artifacts = {}

    for display_name, model_key in models_to_test:
        print(f"\n--- Training {display_name} ---")
        model = train_models(X_train, y_train, model_type=model_key)
        
        if hasattr(model, "predict_proba"):
            val_probs = model.predict_proba(X_val)[:, 1]
        else:
            val_probs = model.predict(X_val)

        val_candidate_scores = [
            (val_pairs_meta[i][0], val_pairs_meta[i][1], float(val_probs[i]))
            for i in range(len(val_pairs_meta))
        ]

        best_t, best_f05, details = optimize_threshold(val_candidate_scores, sub_gt)

        result_row = {
            "Model": display_name,
            "Key": model_key,
            "Best Threshold": best_t,
            "Macro F_0.5": best_f05,
            "Singleton Accuracy": details.get("singleton_accuracy", 1.0) * 100,
            "Non-singleton F_0.5": details.get("non_singleton_f05", 0.0),
        }
        benchmark_results.append(result_row)
        trained_artifacts[model_key] = (model, best_t, best_f05)

    # 5. Model Selection & Comparison Table
    print("\n" + "=" * 80)
    print("MODEL SELECTION & EVALUATION COMPARISON TABLE (LEADERBOARD METRIC: F_0.5)")
    print("=" * 80)
    df_results = pd.DataFrame(benchmark_results)[["Model", "Best Threshold", "Singleton Accuracy", "Non-singleton F_0.5", "Macro F_0.5"]]
    print(df_results.to_string(index=False))

    best_result = max(benchmark_results, key=lambda x: x["Macro F_0.5"])
    champion_key = best_result["Key"]
    champion_model, champion_thresh, champion_score = trained_artifacts[champion_key]

    print("\n" + "=" * 80)
    print(f"CHAMPION MODEL SELECTED: {best_result['Model']}")
    print(f"  Highest Macro F_0.5:   {champion_score:.4f}")
    print(f"  Optimal Threshold:     {champion_thresh:.2f}")
    print("=" * 80)

    champion_path = os.path.join(output_dir, "best_entity_resolver.pkl")
    save_pipeline(champion_model, champion_thresh, champion_path)
    save_pipeline(champion_model, champion_thresh, os.path.join(output_dir, f"entity_resolver_{champion_key}.pkl"))

    print(f"\nChampion model ready for final test inference: {champion_path}")
    return df_results, champion_path


def main():
    parser = argparse.ArgumentParser(description="Benchmark all 4 models and select the highest F_0.5.")
    parser.add_argument("--data-dir", default="train", help="Path to training datasets")
    parser.add_argument("--val-dir", default="data/val", help="Path to validation split")
    parser.add_argument("--n-sample", type=int, default=2000, help="Number of S1 queries to benchmark")
    parser.add_argument("--output-dir", default="models", help="Where to save model artifacts")
    args = parser.parse_args()

    benchmark_all_models(args.data_dir, args.val_dir, n_sample=args.n_sample, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
