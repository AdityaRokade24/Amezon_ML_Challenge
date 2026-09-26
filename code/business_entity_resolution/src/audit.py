#!/usr/bin/env python3
"""
Phase 1: Data Audit & Leakage-Safe Validation Split Generator
=============================================================
Profiles train and test datasets, audits ground truth distribution (singletons vs multi-matches),
and creates an entity-grouped, leakage-safe validation split for reliable F0.5 evaluation.
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from collections import Counter


def audit_file(filepath, name="Dataset"):
    """Profile a single TSV file with streaming/chunking to avoid memory spikes."""
    print(f"\n--- Profiling {name}: {os.path.basename(filepath)} ---")
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found!")
        return None

    total_rows = 0
    missing_counts = Counter()
    countries = Counter()
    cols = None
    sample_records = []

    # Stream file to count and check missing values
    for chunk in pd.read_csv(filepath, sep="\t", chunksize=200000, dtype=str, on_bad_lines='skip'):
        if cols is None:
            cols = list(chunk.columns)
            sample_records = chunk.head(2).to_dict(orient="records")
        total_rows += len(chunk)
        for col in cols:
            missing_counts[col] += chunk[col].isna().sum() + (chunk[col].str.strip() == "").sum()
        if "country" in chunk.columns:
            countries.update(chunk["country"].dropna().str.strip())

    stats = {
        "file": os.path.basename(filepath),
        "total_rows": total_rows,
        "columns": cols,
        "missing_counts": dict(missing_counts),
        "countries": dict(countries),
        "samples": sample_records,
    }

    print(f"  Total Rows: {total_rows:,}")
    print(f"  Columns: {cols}")
    print(f"  Missing fields: {dict(missing_counts)}")
    if countries:
        print(f"  Country Distribution: {dict(countries)}")
    return stats


def audit_ground_truth(gt_path):
    """Profile ground truth to understand singletons, match counts, and source distributions."""
    print(f"\n--- Profiling Ground Truth: {os.path.basename(gt_path)} ---")
    total_s1 = 0
    singletons = 0
    match_counts = Counter()
    s2_matches = 0
    s3_matches = 0

    for chunk in pd.read_csv(gt_path, sep="\t", chunksize=200000, dtype=str, keep_default_na=False):
        total_s1 += len(chunk)
        for _, row in chunk.iterrows():
            raw_matches = row["matched_entity_ids"].strip()
            if not raw_matches:
                singletons += 1
                match_counts[0] += 1
            else:
                m_list = [m.strip() for m in raw_matches.split(",") if m.strip()]
                n = len(m_list)
                match_counts[min(n, 10)] += 1
                for m in m_list:
                    if m.startswith("S2-"):
                        s2_matches += 1
                    elif m.startswith("S3-"):
                        s3_matches += 1

    pct_singleton = (singletons / total_s1) * 100 if total_s1 > 0 else 0
    print(f"  Total S1 Entities in GT: {total_s1:,}")
    print(f"  Singletons (Zero matches): {singletons:,} ({pct_singleton:.2f}%)")
    print(f"  Entities with Matches: {total_s1 - singletons:,} ({100 - pct_singleton:.2f}%)")
    print(f"  Total S2 match links: {s2_matches:,}")
    print(f"  Total S3 match links: {s3_matches:,}")
    print("  Match count distribution per S1 entity:")
    for count in sorted(match_counts.keys()):
        label = f"{count}" if count < 10 else ">=10"
        print(f"    {label} matches: {match_counts[count]:,} entities ({match_counts[count]/total_s1*100:.2f}%)")

    return {
        "total_s1": total_s1,
        "singletons": singletons,
        "singleton_pct": pct_singleton,
        "s2_matches": s2_matches,
        "s3_matches": s3_matches,
        "match_counts": dict(match_counts),
    }


def create_validation_split(data_dir, output_dir, val_size=30000, seed=42):
    """
    Create a leakage-safe validation split.
    Entity groups are kept intact: an S1 entity and ALL its matching S2/S3 records
    are isolated into validation.
    """
    print(f"\n--- Creating Leakage-Safe Validation Split ({val_size:,} S1 entities) ---")
    os.makedirs(output_dir, exist_ok=True)

    gt_path = os.path.join(data_dir, "train", "train_ground_truth.tsv")
    s1_path = os.path.join(data_dir, "train", "train_source1.tsv")

    print("  Sampling S1 entities with country stratification...")
    df_s1 = pd.read_csv(s1_path, sep="\t", dtype=str)
    df_gt = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)

    merged = df_s1.merge(df_gt, left_on="entity_id", right_on="source1_entity_id", how="inner")
    merged["is_singleton"] = merged["matched_entity_ids"].str.strip() == ""

    # Stratified sampling across country and singleton status
    strat_keys = merged["country"].astype(str) + "_" + merged["is_singleton"].astype(str)
    
    np.random.seed(seed)
    sampled_indices = []
    frac = min(1.0, val_size / len(merged))

    for key, group in merged.groupby(strat_keys):
        n_sample = max(1, int(round(len(group) * frac)))
        sampled_indices.extend(group.sample(n=min(n_sample, len(group)), random_state=seed).index)

    val_merged = merged.loc[sampled_indices].copy()
    val_s1_ids = set(val_merged["entity_id"])
    print(f"  Selected {len(val_s1_ids):,} validation S1 entities.")

    # Collect all positive match target IDs
    val_target_ids = set()
    for raw in val_merged["matched_entity_ids"]:
        if raw.strip():
            for mid in raw.split(","):
                val_target_ids.add(mid.strip())
    print(f"  Target positive match IDs in validation: {len(val_target_ids):,}")

    # Save validation ground truth and S1
    val_gt = val_merged[["source1_entity_id", "matched_entity_ids"]]
    val_s1 = val_merged[["entity_id", "business_name", "business_address", "country"]]

    val_gt_path = os.path.join(output_dir, "val_ground_truth.tsv")
    val_s1_path = os.path.join(output_dir, "val_source1.tsv")

    val_gt.to_csv(val_gt_path, sep="\t", index=False)
    val_s1.to_csv(val_s1_path, sep="\t", index=False)
    print(f"  Saved: {val_gt_path}")
    print(f"  Saved: {val_s1_path}")

    # Save metadata summary
    summary = {
        "val_s1_count": len(val_s1_ids),
        "val_target_positive_ids": len(val_target_ids),
        "val_singletons": int((val_merged["is_singleton"]).sum()),
        "val_singleton_pct": float((val_merged["is_singleton"]).mean() * 100),
        "val_country_counts": val_merged["country"].value_counts().to_dict(),
    }
    with open(os.path.join(output_dir, "val_meta.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Validation split summary: {summary}")
    return val_s1_ids, val_target_ids


def main():
    parser = argparse.ArgumentParser(description="Audit data and generate validation split.")
    parser.add_argument("--data-dir", default="6ab10eb3b23ba_student_resource/student_resource/dataset",
                        help="Path to dataset folder containing train/ and test/")
    parser.add_argument("--val-dir", default="data/val",
                        help="Output directory for validation subset")
    parser.add_argument("--val-size", type=int, default=25000,
                        help="Number of S1 entities to include in validation split")
    parser.add_argument("--audit-only", action="store_true", help="Only run audit, skip val creation")
    args = parser.parse_args()

    print("==================================================================")
    print("PHASE 1: DATA AUDIT & LEAKAGE-SAFE VALIDATION SETUP")
    print("==================================================================")

    # 1. Profile Train Datasets
    audit_file(os.path.join(args.data_dir, "train", "train_source1.tsv"), "Train Source 1")
    audit_ground_truth(os.path.join(args.data_dir, "train", "train_ground_truth.tsv"))

    # 2. Profile Test Datasets
    audit_file(os.path.join(args.data_dir, "test", "test_source1.tsv"), "Test Source 1")

    # 3. Create Validation Split
    if not args.audit_only:
        create_validation_split(args.data_dir, args.val_dir, val_size=args.val_size)

    print("\nPhase 1 completed successfully.")


if __name__ == "__main__":
    main()
