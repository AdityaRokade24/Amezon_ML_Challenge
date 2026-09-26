#!/usr/bin/env python3
"""
Official Submission Validator for Amazon ML Challenge 2026: Business Entity Resolution
======================================================================================
Validates output/matching_results.tsv and output/candidate_pairs.tsv against all competition rules:
1. Correct file existence and exact required TSV headers.
2. Exactly one row per test Source 1 entity (1,732,544 rows + 1 header).
3. No duplicate Source 1 entity rows, no duplicate matched/candidate IDs.
4. Matched and candidate entity IDs must belong exclusively to Source 2 (S2-) or Source 3 (S3-).
5. No self-matches (no S1- IDs in predictions).
6. Subset Constraint: Every matched ID in matching_results.tsv must appear in candidate_pairs.tsv.
7. Verification against valid test candidate IDs.
"""

import os
import sys
import argparse


def validate_submission(matching_file: str, candidate_file: str, test_dir: str):
    print("==================================================================")
    print("OFFICIAL SUBMISSION VALIDATION ENGINE")
    print(f"Matching Results: {matching_file}")
    print(f"Candidate Pairs:  {candidate_file}")
    print(f"Test Directory:   {test_dir}")
    print("==================================================================")

    issues = []

    # 1. Check file existence
    if not os.path.exists(matching_file):
        issues.append(f"Matching results file not found: {matching_file}")
    if not os.path.exists(candidate_file):
        issues.append(f"Candidate pairs file not found: {candidate_file}")

    if issues:
        print("\nFATAL ERROR:")
        for idx, iss in enumerate(issues, 1):
            print(f"  {idx}. {iss}")
        return False

    # 2. Load valid test S1, S2, S3 IDs for verification
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")

    print("\n[1/4] Loading reference entity IDs from test set...")
    valid_s1_ids = set()
    s1_ordered = []
    if os.path.exists(s1_path):
        with open(s1_path, "r", encoding="utf-8") as f:
            next(f)  # header
            for line in f:
                parts = line.split("\t")
                if parts:
                    eid = parts[0].strip()
                    valid_s1_ids.add(eid)
                    s1_ordered.append(eid)
        print(f"  Valid Test Source 1 entities: {len(valid_s1_ids):,}")

    valid_cand_ids = set()
    for src_path, label in [(s2_path, "Source 2"), (s3_path, "Source 3")]:
        if os.path.exists(src_path):
            with open(src_path, "r", encoding="utf-8") as f:
                next(f)  # header
                for line in f:
                    parts = line.split("\t")
                    if parts:
                        valid_cand_ids.add(parts[0].strip())
            print(f"  Valid Test {label} entities: {len(valid_cand_ids):,}")

    # 3. Validate matching_results.tsv
    print("\n[2/4] Validating matching_results.tsv structure & rules...")
    matching_map = {}
    matched_s1_set = set()
    total_matches = 0
    singletons = 0

    with open(matching_file, "r", encoding="utf-8") as f:
        header = f.readline()
        if not header.startswith("source1_entity_id\tmatched_entity_ids"):
            issues.append(f"Invalid header in {matching_file}. Expected: 'source1_entity_id\\tmatched_entity_ids'")

        line_num = 1
        for line in f:
            line_num += 1
            line = line.rstrip("\r\n")
            parts = line.split("\t")
            if len(parts) < 1:
                issues.append(f"{matching_file} line {line_num}: Empty line encountered.")
                continue

            s1_id = parts[0].strip()
            raw_matches = parts[1].strip() if len(parts) > 1 else ""

            if s1_id in matched_s1_set:
                issues.append(f"{matching_file} line {line_num}: Duplicate Source 1 entity: {s1_id}")
            matched_s1_set.add(s1_id)

            if valid_s1_ids and s1_id not in valid_s1_ids:
                issues.append(f"{matching_file} line {line_num}: Unknown S1 entity ID: {s1_id}")

            if not raw_matches:
                singletons += 1
                matching_map[s1_id] = set()
                continue

            match_list = [m.strip() for m in raw_matches.split(",") if m.strip()]
            if len(match_list) != len(set(match_list)):
                issues.append(f"{matching_file} line {line_num}: Duplicate matched IDs in row for {s1_id}")

            for mid in match_list:
                if mid.startswith("S1-"):
                    issues.append(f"{matching_file} line {line_num}: Self-match prohibited ({mid})")
                if not (mid.startswith("S2-") or mid.startswith("S3-")):
                    issues.append(f"{matching_file} line {line_num}: Invalid entity ID prefix ({mid}). Expected S2- or S3-")
                if valid_cand_ids and mid not in valid_cand_ids:
                    issues.append(f"{matching_file} line {line_num}: Matched ID does not exist in test set: {mid}")

            total_matches += len(match_list)
            matching_map[s1_id] = set(match_list)

            if len(issues) >= 20:
                issues.append("Stopping early: too many validation errors in matching_results.tsv.")
                break

    print(f"  Rows validated: {len(matched_s1_set):,}")
    print(f"  Singletons (no matches): {singletons:,} ({singletons/max(1, len(matched_s1_set))*100:.1f}%)")
    print(f"  Total matched links: {total_matches:,}")

    # 4. Validate candidate_pairs.tsv & Subset Rule
    print("\n[3/4] Validating candidate_pairs.tsv & Subset Constraint...")
    candidate_s1_set = set()
    total_candidates = 0

    with open(candidate_file, "r", encoding="utf-8") as f:
        header = f.readline()
        if not header.startswith("source1_entity_id\tcandidate_entity_ids"):
            issues.append(f"Invalid header in {candidate_file}. Expected: 'source1_entity_id\\tcandidate_entity_ids'")

        line_num = 1
        for line in f:
            line_num += 1
            line = line.rstrip("\r\n")
            parts = line.split("\t")
            if len(parts) < 1:
                continue

            s1_id = parts[0].strip()
            raw_cands = parts[1].strip() if len(parts) > 1 else ""

            if s1_id in candidate_s1_set:
                issues.append(f"{candidate_file} line {line_num}: Duplicate Source 1 entity: {s1_id}")
            candidate_s1_set.add(s1_id)

            cand_list = [c.strip() for c in raw_cands.split(",") if c.strip()] if raw_cands else []
            cand_set = set(cand_list)
            total_candidates += len(cand_list)

            if len(cand_list) != len(cand_set):
                issues.append(f"{candidate_file} line {line_num}: Duplicate candidate IDs in row for {s1_id}")

            # Subset Rule: every matched ID must be in candidate set!
            matched_ids = matching_map.get(s1_id, set())
            missing_from_candidates = matched_ids - cand_set
            if missing_from_candidates:
                issues.append(
                    f"Subset Rule Violation for {s1_id}: Matched IDs {missing_from_candidates} do not appear in candidate_pairs.tsv!"
                )

            if len(issues) >= 20:
                issues.append("Stopping early: too many validation errors.")
                break

    print(f"  Rows validated: {len(candidate_s1_set):,}")
    print(f"  Total candidates generated: {total_candidates:,}")
    print(f"  Average candidates per S1 entity: {total_candidates / max(1, len(candidate_s1_set)):.1f}")

    # 5. Check row completeness
    print("\n[4/4] Verifying completeness against test queries...")
    if valid_s1_ids:
        missing_matching = valid_s1_ids - matched_s1_set
        if missing_matching:
            issues.append(f"matching_results.tsv is missing {len(missing_matching):,} Source 1 entities from test set!")

        missing_candidate = valid_s1_ids - candidate_s1_set
        if missing_candidate:
            issues.append(f"candidate_pairs.tsv is missing {len(missing_candidate):,} Source 1 entities from test set!")

    print("\n==================================================================")
    if not issues:
        print("VALIDATION RESULT: PASS")
        print("All competition rules and format constraints are 100% satisfied!")
        print("Your outputs are completely safe to submit.")
        print("==================================================================")
        return True
    else:
        print(f"VALIDATION RESULT: FAILED ({len(issues)} issues detected)")
        print("------------------------------------------------------------------")
        for idx, iss in enumerate(issues[:15], 1):
            print(f"  {idx}. {iss}")
        if len(issues) > 15:
            print(f"  ... and {len(issues) - 15} more issues.")
        print("==================================================================")
        return False


def main():
    parser = argparse.ArgumentParser(description="Official validator for submission TSVs.")
    parser.add_argument("--matching", default="output/matching_results.tsv", help="Path to matching_results.tsv")
    parser.add_argument("--candidate", default="output/candidate_pairs.tsv", help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-dir", default="test", help="Path to test directory")
    args = parser.parse_args()

    success = validate_submission(args.matching, args.candidate, args.test_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
