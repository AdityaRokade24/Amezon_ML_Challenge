#!/usr/bin/env python3
"""
Phase 8: Submission Validator & Automated Packaging Engine
==========================================================
1. Runs the official validation check via validate_submission.py.
2. Validates that all required files and directory structures exist.
3. Assembles the final zip archive matching the exact required competition structure:
   <team_name>_submission.zip
   ├── output/
   │   ├── matching_results.tsv
   │   └── candidate_pairs.tsv
   ├── code/
   │   └── business_entity_resolution/
   │       ├── src/
   │       ├── README.md
   │       └── requirements.txt
   └── Documentation_template.md
"""

import os
import sys
import zipfile
import subprocess
import argparse


def run_official_validator(matching_path: str, candidate_path: str, test_dir: str):
    """Execute official validate_submission.py before packaging."""
    validator_candidates = [
        "utils/validate_submission.py",
        "6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py"
    ]
    validator_path = None
    for vp in validator_candidates:
        if os.path.exists(vp):
            validator_path = vp
            break

    if not validator_path:
        print("Warning: Validator script not found.")
        return True

    cmd = [
        sys.executable,
        validator_path,
        "--matching", matching_path,
        "--candidate", candidate_path,
        "--test-dir", test_dir
    ]
    print(f"Running official submission validator:\n  {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print(res.stderr)
    return res.returncode == 0


def create_submission_zip(team_name: str, root_dir: str = "."):
    """Bundle the official submission zip according to exact specification."""
    zip_filename = f"{team_name}_submission.zip"
    zip_filepath = os.path.join(root_dir, zip_filename)

    required_items = [
        ("output/matching_results.tsv", "output/matching_results.tsv"),
        ("output/candidate_pairs.tsv", "output/candidate_pairs.tsv"),
        ("Documentation_template.md", "Documentation_template.md"),
        ("code/business_entity_resolution/requirements.txt", "code/business_entity_resolution/requirements.txt"),
        ("code/business_entity_resolution/README.md", "code/business_entity_resolution/README.md"),
    ]

    # Check existence
    missing = []
    for src_rel, _ in required_items:
        full_src = os.path.join(root_dir, src_rel)
        if not os.path.exists(full_src):
            missing.append(src_rel)

    if missing:
        print(f"Error: Missing required files for packaging:\n  " + "\n  ".join(missing))
        return False

    print(f"\nBuilding submission zip: {zip_filename}...")
    with zipfile.ZipFile(zip_filepath, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add required individual files
        for src_rel, arc_path in required_items:
            full_src = os.path.join(root_dir, src_rel)
            zf.write(full_src, arc_path)
            print(f"  Added: {arc_path}")

        # Add all source code in code/business_entity_resolution/src/
        src_dir = os.path.join(root_dir, "code", "business_entity_resolution", "src")
        for root, _, files in os.walk(src_dir):
            for file in files:
                if file.endswith((".py", ".sh", ".json")):
                    full_p = os.path.join(root, file)
                    rel_p = os.path.relpath(full_p, root_dir)
                    zf.write(full_p, rel_p)
                    print(f"  Added: {rel_p}")

    print(f"\nPackage created successfully: {zip_filepath}")
    print(f"Archive Size: {os.path.getsize(zip_filepath) / (1024 * 1024):.2f} MB")
    return True


def main():
    parser = argparse.ArgumentParser(description="Validate and package the final competition submission.")
    parser.add_argument("--team-name", default="Team_Entity_Resolver", help="Team name for zip filename")
    parser.add_argument("--test-dir", default="6ab10eb3b23ba_student_resource/student_resource/dataset/test",
                        help="Path to test dataset")
    parser.add_argument("--skip-validator", action="store_true", help="Skip running validator before packaging")
    args = parser.parse_args()

    matching = "output/matching_results.tsv"
    candidate = "output/candidate_pairs.tsv"

    if not args.skip_validator and os.path.exists(matching):
        valid = run_official_validator(matching, candidate, args.test_dir)
        if not valid:
            print("Validation FAILED! Please resolve the formatting errors before packaging.")
            sys.exit(1)

    success = create_submission_zip(args.team_name)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
