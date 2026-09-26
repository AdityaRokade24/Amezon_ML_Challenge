#!/usr/bin/env python3
"""
Official Macro F_0.5 Score Calculator for Amazon ML Challenge 2026
==================================================================
Streams train_result.tsv and train_ground_truth.tsv to compute the exact official score:
Formula: F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
- Evaluates Macro-Average across all 2,206,821 Source 1 entities.
- Full credit (1.0) for correctly identified singletons (predicted empty == true empty).
- Heavy penalty (0.0) for false merges on singletons.
"""

import os
import sys
import time
import argparse


def compute_entity_f05(pred_set: set, true_set: set) -> float:
    n_true = len(true_set)
    n_pred = len(pred_set)

    if n_true == 0:
        return 1.0 if n_pred == 0 else 0.0

    if n_pred == 0:
        return 0.0

    tp = len(pred_set.intersection(true_set))
    if tp == 0:
        return 0.0

    precision = tp / n_pred
    recall = tp / n_true
    denom = 0.25 * precision + recall
    return (1.25 * precision * recall) / denom if denom > 0 else 0.0


def score_predictions(pred_path: str = "outputs/train_result.tsv",
                      gt_path: str = "train/train_ground_truth.tsv"):
    print("==================================================================")
    print("OFFICIAL MACRO F_0.5 SCORE EVALUATION")
    print(f"Predictions:  {pred_path}")
    print(f"Ground Truth: {gt_path}")
    print("==================================================================")

    if not os.path.exists(pred_path):
        print(f"Error: Predictions file not found: {pred_path}")
        return
    if not os.path.exists(gt_path):
        # Check alternative locations
        for alt in ["dataset/train/train_ground_truth.tsv", "data/val/val_ground_truth.tsv"]:
            if os.path.exists(alt):
                gt_path = alt
                break

    print("\n[1/2] Loading ground truth mappings...")
    t0 = time.time()
    gt_map = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("\t")
            s1_id = parts[0].strip()
            raw_matches = parts[1].strip() if len(parts) > 1 else ""
            gt_map[s1_id] = set(m.strip() for m in raw_matches.split(",") if m.strip()) if raw_matches else set()

    print(f"  Loaded {len(gt_map):,} ground truth entities in {time.time()-t0:.2f}s.")

    print("\n[2/2] Streaming predictions and computing Macro F_0.5...")
    t1 = time.time()
    total_evaluated = 0
    total_score = 0.0
    perfect_matches = 0
    singleton_correct = 0
    singleton_false_merges = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    with open(pred_path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("\t")
            s1_id = parts[0].strip()
            raw_matches = parts[1].strip() if len(parts) > 1 else ""
            pred_set = set(m.strip() for m in raw_matches.split(",") if m.strip()) if raw_matches else set()

            if s1_id in gt_map:
                true_set = gt_map[s1_id]
                score = compute_entity_f05(pred_set, true_set)
                total_score += score
                total_evaluated += 1

                # Statistics
                tp = len(pred_set.intersection(true_set))
                fp = len(pred_set - true_set)
                fn = len(true_set - pred_set)
                total_tp += tp
                total_fp += fp
                total_fn += fn

                if len(true_set) == 0:
                    if len(pred_set) == 0:
                        singleton_correct += 1
                    else:
                        singleton_false_merges += 1
                elif pred_set == true_set:
                    perfect_matches += 1

                if total_evaluated % 500000 == 0:
                    running_f05 = total_score / total_evaluated
                    print(f"  Scored {total_evaluated:,} entities... Running Macro F_0.5: {running_f05:.4f}")

    if total_evaluated == 0:
        print("Error: No matching Source 1 entities found between prediction and ground truth.")
        return

    macro_f05 = total_score / total_evaluated
    overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    print(f"\nCompleted in {time.time()-t1:.2f}s ({total_evaluated/max(0.001, time.time()-t1):,.0f} entities/sec).")
    print("\n==================================================================")
    print("FINAL OFFICIAL EVALUATION RESULTS")
    print("==================================================================")
    print(f"  Total Entities Evaluated:      {total_evaluated:,}")
    print(f"  ★ MACRO F_0.5 SCORE:          {macro_f05:.4f} ({macro_f05*100:.2f}%)")
    print(f"  Pairwise Precision:            {overall_p:.4f} ({overall_p*100:.2f}%)")
    print(f"  Pairwise Recall:               {overall_r:.4f} ({overall_r*100:.2f}%)")
    print(f"  Perfect Full Matches (1.0):    {perfect_matches:,} ({perfect_matches/total_evaluated*100:.2f}%)")
    print(f"  Singletons Correctly Kept:     {singleton_correct:,}")
    print(f"  Singleton False Merges:        {singleton_false_merges:,}")
    print("==================================================================")


def main():
    parser = argparse.ArgumentParser(description="Official Macro F_0.5 Score Calculator")
    parser.add_argument("--pred", default="outputs/train_result.tsv", help="Path to predictions TSV")
    parser.add_argument("--gt", default="train/train_ground_truth.tsv", help="Path to ground truth TSV")
    args = parser.parse_args()

    score_predictions(args.pred, args.gt)


if __name__ == "__main__":
    main()
