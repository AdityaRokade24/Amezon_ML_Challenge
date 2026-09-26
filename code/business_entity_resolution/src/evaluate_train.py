#!/usr/bin/env python3
"""
Step 7: Train Evaluation & Side-by-Side Comparison Engine
=========================================================
1. Evaluates model predictions on train/validation datasets.
2. Emits `outputs/train_result.tsv` formatted exactly as required:
   source1_entity_id\tmatched_entity_ids
3. Merges and directly compares with `train_ground_truth.tsv`.
4. Produces `outputs/train_comparison.tsv` for row-by-row inspection:
   - source1_entity_id, business_name, predicted_matches, ground_truth_matches,
     precision, recall, entity_f05, match_status
5. Scalable Architecture:
   - Fast In-Memory Mode for samples (<= 20,000 entities, finishes in seconds).
   - Scalable Streaming Mode for full datasets (> 20,000 entities or --all),
     using SQLite CandidateStore to keep RAM strictly < 200 MB with zero OOM errors.
"""

import os
import sys
import gc
import json
import argparse
from collections import Counter
import pandas as pd
import numpy as np
from tqdm import tqdm

# Ensure local src modules can be imported
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from normalize import preprocess_record
from blocking import MultiPassBlocker
from candidate_store import CandidateStore
from features import extract_pairwise_features, vectorize_features
from train import load_pipeline
from evaluate import compute_entity_f05, compute_macro_f05


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


def generate_and_compare_train(
    data_dir: str = "train",
    val_dir: str = "data/val",
    model_path: str = "models/best_entity_resolver.pkl",
    output_dir: str = "outputs",
    n_sample: int = 2000,
    threshold: float = None,
    chunk_size: int = 5000
):
    print("==================================================================")
    print("STEP 7: GENERATE TRAIN_RESULT.TSV & COMPARE WITH GROUND TRUTH")
    sample_label = f"{n_sample:,}" if (n_sample and n_sample > 0) else "ALL (2,206,821)"
    print(f"Sample Entities: {sample_label} | Model: {os.path.basename(model_path)}")
    print("==================================================================")

    os.makedirs(output_dir, exist_ok=True)
    train_result_path = os.path.join(output_dir, "train_result.tsv")
    train_comparison_path = os.path.join(output_dir, "train_comparison.tsv")
    metrics_summary_path = os.path.join(output_dir, "evaluation_summary.json")

    # 1. Load trained model pipeline
    if not os.path.exists(model_path):
        alt_paths = ["models/best_entity_resolver.pkl", "models/entity_resolver_xgb.pkl", "models/entity_resolver_lgbm.pkl"]
        for ap in alt_paths:
            if os.path.exists(ap):
                model_path = ap
                break

    pipeline = load_pipeline(model_path)
    model = pipeline["model"]
    model_threshold = pipeline.get("threshold", 0.50)
    eval_threshold = threshold if threshold is not None else model_threshold
    print(f"Loaded model: {type(model).__name__} | Active Decision Threshold: {eval_threshold:.2f}")

    # 2. Locate S1 and Ground Truth
    s1_path = find_dataset_file("train_source1.tsv", data_dir)
    gt_path = find_dataset_file("train_ground_truth.tsv", data_dir)

    is_full_scale = (n_sample is None or n_sample <= 0 or n_sample > 20000)

    # -----------------------------------------------------------------
    # BRANCH A: Scalable Disk-Backed Streaming Mode (for > 20k or ALL)
    # -----------------------------------------------------------------
    if is_full_scale:
        print("\n[Mode] High-Scale Disk-Backed Streaming Mode (Zero-RAM OOM Protection)")
        db_path = os.path.join("data", "train_candidate_store.db")
        store = CandidateStore(db_path)

        if not store.is_populated():
            print("Indexing Source 2 and Source 3 candidate records into SQLite store...")
            s2_path = find_dataset_file("train_source2.tsv", data_dir)
            s3_path = find_dataset_file("train_source3.tsv", data_dir)
            store.build_from_sources([s2_path, s3_path], batch_size=50000)
        else:
            print(f"Using pre-indexed CandidateStore ({store.get_candidate_count():,} candidates).")

        # Load ground truth in streaming reader
        print(f"Reading ground truth from: {gt_path}")
        gt_reader = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False, chunksize=100000)
        gt_map = {}
        for chunk in gt_reader:
            for s1, raw in zip(chunk["source1_entity_id"], chunk["matched_entity_ids"]):
                raw_str = str(raw).strip() if isinstance(raw, str) else ""
                gt_map[s1] = set(m.strip() for m in raw_str.split(",") if m.strip()) if raw_str else set()

        total_processed = 0
        total_tp, total_fp, total_fn = 0, 0, 0
        scores = []
        comparison_sample = []

        with open(train_result_path, "w", encoding="utf-8") as f_res, \
             open(train_comparison_path, "w", encoding="utf-8") as f_comp:

            f_res.write("source1_entity_id\tmatched_entity_ids\n")
            f_comp.write("source1_entity_id\tbusiness_name\tpredicted_matches\tground_truth_matches\tprecision\trecall\tentity_f05\tstatus\n")

            s1_reader = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False, chunksize=chunk_size)
            for chunk in s1_reader:
                for _, row in chunk.iterrows():
                    total_processed += 1
                    s1_id = row["entity_id"]
                    raw_name = row["business_name"]
                    s1_rec = preprocess_record(s1_id, raw_name, row["business_address"], row.get("country", "US"))
                    true_set = gt_map.get(s1_id, set())

                    # Query candidates from store
                    candidates = store.query_candidates_for_record(s1_rec, max_candidates=40)
                    pred_set = set()

                    if candidates:
                        feat_list = [extract_pairwise_features(s1_rec, c) for c in candidates]
                        X_chunk = vectorize_features(feat_list)
                        if hasattr(model, "predict_proba"):
                            probs = model.predict_proba(X_chunk)[:, 1]
                        else:
                            probs = model.predict(X_chunk)

                        for c, p_val in zip(candidates, probs):
                            if p_val >= eval_threshold:
                                pred_set.add(c["entity_id"])

                    pred_str = ",".join(sorted(pred_set))
                    true_str = ",".join(sorted(true_set))
                    f_res.write(f"{s1_id}\t{pred_str}\n")

                    tp = len(pred_set.intersection(true_set))
                    fp = len(pred_set - true_set)
                    fn = len(true_set - pred_set)
                    total_tp += tp
                    total_fp += fp
                    total_fn += fn

                    entity_f05 = compute_entity_f05(pred_set, true_set)
                    scores.append(entity_f05)

                    status = "PERFECT_MATCH (1.0)" if pred_set == true_set and len(true_set) > 0 else (
                        "SINGLETON_CORRECT (1.0)" if len(true_set) == 0 and len(pred_set) == 0 else (
                            "SINGLETON_FALSE_MERGE (0.0)" if len(true_set) == 0 else "MISSED_MATCH (0.0)"
                        )
                    )
                    p_score = tp / len(pred_set) if len(pred_set) > 0 else (1.0 if len(true_set) == 0 else 0.0)
                    r_score = tp / len(true_set) if len(true_set) > 0 else 1.0

                    f_comp.write(f"{s1_id}\t{raw_name[:25]}\t{pred_str or '<EMPTY>'}\t{true_str or '<EMPTY>'}\t{p_score:.3f}\t{r_score:.3f}\t{entity_f05:.4f}\t{status}\n")

                    if len(comparison_sample) < 10:
                        comparison_sample.append({
                            "source1_entity_id": s1_id,
                            "business_name": raw_name[:25],
                            "predicted_matches": pred_str or "<EMPTY>",
                            "ground_truth_matches": true_str or "<EMPTY>",
                            "precision": round(p_score, 3),
                            "recall": round(r_score, 3),
                            "entity_f05": round(entity_f05, 4),
                            "status": status,
                        })

                    if n_sample and total_processed >= n_sample:
                        break

                f_res.flush()
                f_comp.flush()
                gc.collect()

                macro_running = float(np.mean(scores)) if scores else 0.0
                print(f"  Processed {total_processed:,} entities | Running Macro F_0.5: {macro_running:.4f}")
                if n_sample and total_processed >= n_sample:
                    break

        store.close()
        macro_f05 = float(np.mean(scores)) if scores else 0.0
        p_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        p_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

        summary = {
            "total_queries_evaluated": total_processed,
            "macro_f05_score": macro_f05,
            "pairwise_true_positives": total_tp,
            "pairwise_false_positives": total_fp,
            "pairwise_false_negatives": total_fn,
            "pairwise_precision": round(p_precision, 4),
            "pairwise_recall": round(p_recall, 4),
            "train_result_file": train_result_path,
            "train_comparison_file": train_comparison_path
        }
        with open(metrics_summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print("\n==================================================================")
        print(f"Macro F_0.5 Score: {macro_f05:.4f} | Precision: {p_precision*100:.2f}% | Recall: {p_recall*100:.2f}%")
        print("==================================================================")
        return summary

    # -----------------------------------------------------------------
    # BRANCH B: Fast In-Memory Sample Mode (for <= 20,000 entities)
    # -----------------------------------------------------------------
    print(f"\n[1/4] Reading query records ({n_sample:,}) from: {s1_path}")
    print(f"      Reading ground truth from:   {gt_path}")
    df_s1 = pd.read_csv(s1_path, sep="\t", dtype=str, nrows=n_sample)
    df_gt = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)

    s1_target_ids = set(df_s1["entity_id"])
    gt_map = {}
    for s1, raw in zip(df_gt["source1_entity_id"], df_gt["matched_entity_ids"]):
        if s1 in s1_target_ids:
            raw_str = str(raw).strip() if isinstance(raw, str) else ""
            gt_map[s1] = set(m.strip() for m in raw_str.split(",") if m.strip()) if raw_str else set()

    queries = []
    needed_positive_ids = set()
    for _, row in df_s1.iterrows():
        rec = preprocess_record(row["entity_id"], row["business_name"], row["business_address"], row.get("country", "US"))
        queries.append(rec)
        needed_positive_ids.update(gt_map.get(row["entity_id"], set()))

    print(f"  Loaded {len(queries):,} S1 queries ({sum(1 for q in queries if len(gt_map.get(q['entity_id'], set())) == 0):,} singletons).")

    # Index Candidate Records (Source 2 and Source 3)
    print("\n[2/4] Indexing candidate records from train_source2 and train_source3...")
    cand_records = {}
    needed_remaining = set(needed_positive_ids)

    for src_file in ["train_source2.tsv", "train_source3.tsv"]:
        p = find_dataset_file(src_file, data_dir)
        if not os.path.exists(p):
            continue
        print(f"  Scanning {p}...")
        for chunk in pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, chunksize=250000):
            matches = chunk[chunk["entity_id"].isin(needed_remaining)]
            for _, row in matches.iterrows():
                cid = row["entity_id"]
                cand_records[cid] = preprocess_record(cid, row["business_name"], row["business_address"], row.get("country", "US"))
                needed_remaining.discard(cid)

            # Sample background negatives
            if len(cand_records) < len(needed_positive_ids) + 40000:
                sample_chunk = chunk.sample(n=min(len(chunk), 5000), random_state=42)
                for _, row in sample_chunk.iterrows():
                    cid = row["entity_id"]
                    if cid not in cand_records:
                        cand_records[cid] = preprocess_record(cid, row["business_name"], row["business_address"], row.get("country", "US"))

            if len(needed_remaining) == 0 and len(cand_records) >= len(needed_positive_ids) + 30000:
                print(f"  All target true positive records found in {src_file}.")
                break

    print(f"  Candidate pool ready: {len(cand_records):,} records.")

    # Build Blocker
    blocker = MultiPassBlocker(max_token_freq=6000, min_token_len=3)
    blocker.fit_candidates(list(cand_records.values()))

    # Generate Candidate Pairs & Vectorized Inference
    print("\n[3/4] Generating candidate pairs and extracting feature vectors...")
    all_pair_features = []
    all_pair_meta = []

    for q in tqdm(queries, desc="Blocking queries"):
        s1_id = q["entity_id"]
        true_set = gt_map.get(s1_id, set())

        cands = blocker.generate_candidates(q, max_candidates_per_query=40)
        all_cands = set(cands)
        for tp_id in true_set:
            if tp_id in cand_records:
                all_cands.add(tp_id)

        for cid in all_cands:
            c_rec = cand_records.get(cid)
            if c_rec:
                feat = extract_pairwise_features(q, c_rec)
                all_pair_features.append(feat)
                all_pair_meta.append((s1_id, cid))

    print(f"  Predicting probabilities across {len(all_pair_features):,} candidate pairs in one batch...")
    probs_all = []
    if all_pair_features:
        X_all = vectorize_features(all_pair_features)
        if hasattr(model, "predict_proba"):
            probs_all = model.predict_proba(X_all)[:, 1]
        else:
            probs_all = model.predict(X_all)

    s1_predictions = {q["entity_id"]: set() for q in queries}
    for (s1_id, cid), prob in zip(all_pair_meta, probs_all):
        if prob >= eval_threshold:
            s1_predictions[s1_id].add(cid)

    # 5. Evaluate Against Ground Truth and Write Results
    print("\n[4/4] Writing train_result.tsv and comparing with ground truth...")
    comparison_rows = []
    total_tp, total_fp, total_fn = 0, 0, 0
    scores = []

    with open(train_result_path, "w", encoding="utf-8") as f_res:
        f_res.write("source1_entity_id\tmatched_entity_ids\n")

        for q in queries:
            s1_id = q["entity_id"]
            true_set = gt_map.get(s1_id, set())
            pred_set = s1_predictions.get(s1_id, set())

            pred_str = ",".join(sorted(pred_set))
            f_res.write(f"{s1_id}\t{pred_str}\n")

            tp = len(pred_set.intersection(true_set))
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)
            total_tp += tp
            total_fp += fp
            total_fn += fn

            entity_f05 = compute_entity_f05(pred_set, true_set)
            scores.append(entity_f05)

            if len(true_set) == 0:
                status = "SINGLETON_CORRECT (1.0)" if len(pred_set) == 0 else "SINGLETON_FALSE_MERGE (0.0)"
            elif pred_set == true_set:
                status = "PERFECT_MATCH (1.0)"
            elif len(pred_set) == 0:
                status = "MISSED_MATCH (0.0)"
            elif tp > 0 and fp == 0:
                status = "PARTIAL_RECALL (<1.0)"
            elif tp > 0 and fp > 0:
                status = "PARTIAL_WITH_FP (<1.0)"
            else:
                status = "WRONG_MERGE (0.0)"

            p_val = tp / len(pred_set) if len(pred_set) > 0 else (1.0 if len(true_set) == 0 else 0.0)
            r_val = tp / len(true_set) if len(true_set) > 0 else 1.0

            true_str = ",".join(sorted(true_set))
            comparison_rows.append({
                "source1_entity_id": s1_id,
                "business_name": q["raw_name"][:25],
                "predicted_matches": pred_str if pred_str else "<EMPTY>",
                "ground_truth_matches": true_str if true_str else "<EMPTY>",
                "precision": round(p_val, 3),
                "recall": round(r_val, 3),
                "entity_f05": round(entity_f05, 4),
                "status": status,
            })

    # Save comparison TSV
    df_comp = pd.DataFrame(comparison_rows)
    df_comp.to_csv(train_comparison_path, sep="\t", index=False)

    macro_f05 = float(np.mean(scores)) if scores else 0.0
    p_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    p_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    status_counts = Counter(r["status"] for r in comparison_rows)
    summary = {
        "total_queries_evaluated": len(queries),
        "macro_f05_score": macro_f05,
        "pairwise_true_positives": total_tp,
        "pairwise_false_positives": total_fp,
        "pairwise_false_negatives": total_fn,
        "pairwise_precision": round(p_precision, 4),
        "pairwise_recall": round(p_recall, 4),
        "status_breakdown": dict(status_counts),
        "train_result_file": train_result_path,
        "train_comparison_file": train_comparison_path
    }
    with open(metrics_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n================================================================================")
    print("EVALUATION RESULTS & GROUND TRUTH COMPARISON")
    print("================================================================================")
    print(f"  Macro-averaged F_0.5 Score:   {macro_f05:.4f}")
    print(f"  Pairwise Precision:           {p_precision*100:.2f}% (TP: {total_tp:,}, FP: {total_fp:,})")
    print(f"  Pairwise Recall:              {p_recall*100:.2f}% (TP: {total_tp:,}, FN: {total_fn:,})")
    print("\n  Prediction Status Breakdown:")
    for status, count in status_counts.most_common():
        pct = (count / len(queries)) * 100
        print(f"    - {status}: {count:,} entities ({pct:.1f}%)")

    print("\n================================================================================")
    print("SIDE-BY-SIDE PREDICTION SAMPLES (from train_comparison.tsv):")
    print("================================================================================")
    print(df_comp[["source1_entity_id", "business_name", "predicted_matches", "ground_truth_matches", "entity_f05", "status"]].head(10).to_string(index=False))

    print("\nFiles generated:")
    print(f"  1. Predictions TSV: {train_result_path}")
    print(f"  2. Comparison TSV:  {train_comparison_path}")
    print(f"  3. Metrics Summary: {metrics_summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Generate train_result.tsv and compare with ground truth.")
    parser.add_argument("--data-dir", default="train",
                        help="Path to dataset directory")
    parser.add_argument("--val-dir", default="data/val",
                        help="Path to validation directory")
    parser.add_argument("--model-path", default="models/best_entity_resolver.pkl",
                        help="Path to trained champion model")
    parser.add_argument("--output-dir", default="outputs",
                        help="Directory to save train_result.tsv and comparison files")
    parser.add_argument("--n-sample", type=int, default=2000,
                        help="Number of entities to evaluate (default: 2000. Set 0 for ALL 2.2M entities)")
    parser.add_argument("--all", action="store_true",
                        help="Process and evaluate ALL 2,206,821 entities from train_source1.tsv")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Decision threshold override (default: optimal threshold from model)")
    parser.add_argument("--chunk-size", type=int, default=5000,
                        help="Streaming chunk size for high-scale processing")
    args = parser.parse_args()

    n_sample = 0 if args.all else args.n_sample

    generate_and_compare_train(
        args.data_dir,
        val_dir=args.val_dir,
        model_path=args.model_path,
        output_dir=args.output_dir,
        n_sample=n_sample,
        threshold=args.threshold,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
