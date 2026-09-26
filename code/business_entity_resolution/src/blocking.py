#!/usr/bin/env python3
"""
Phase 3: Multi-Pass High-Recall Blocking Engine
================================================
Implements:
1. Token-based inverted index (business name & address tokens)
2. Character 3-gram index for typo and spelling error recovery
3. Frequency-weighted token pruning (removing ubiquitous generic stopwords)
4. Candidate union and metrics computation (Candidate Recall & Reduction Ratio)
"""

import math
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple


COMMON_BLOCKING_STOPWORDS = {
    "the", "and", "or", "in", "at", "of", "to", "for", "with", "a", "an",
    "co", "company", "inc", "incorporated", "ltd", "limited", "pvt", "corp",
    "corporation", "llc", "llp", "enterprises", "services", "solutions",
    "group", "holdings", "management", "international", "global", "industries",
    "street", "road", "rd", "st", "ave", "avenue", "dr", "drive", "suite",
    "floor", "building", "box", "po", "near", "opp", "opposite", "usa", "india", "france"
}


class MultiPassBlocker:
    """Multi-pass candidate generator combining token inverted index and n-gram overlap."""

    def __init__(self, max_token_freq: int = 5000, min_token_len: int = 3):
        self.max_token_freq = max_token_freq
        self.min_token_len = min_token_len
        self.token_index = defaultdict(list)
        self.ngram_index = defaultdict(list)
        self.record_lookup = {}
        self.token_counts = Counter()

    def fit_candidates(self, candidate_records: List[dict]):
        """
        Build inverted indexes over candidate records (S2 and S3).
        Each candidate_record is a dict from normalize.preprocess_record.
        """
        print(f"Building blocking indexes over {len(candidate_records):,} candidate records...")
        
        # Pass 1: Count token frequencies
        for rec in candidate_records:
            cid = rec["entity_id"]
            self.record_lookup[cid] = rec
            # Name tokens
            for tok in rec["name_tokens"]:
                if len(tok) >= self.min_token_len and tok not in COMMON_BLOCKING_STOPWORDS:
                    self.token_counts[tok] += 1
            # Address tokens
            for atok in rec.get("addr_tokens", []):
                if len(atok) >= self.min_token_len and atok not in COMMON_BLOCKING_STOPWORDS:
                    self.token_counts[f"ADDR:{atok}"] += 1
            # Postal code
            if rec.get("postal_code"):
                self.token_counts[rec["postal_code"]] += 1

        # Pass 2: Populate inverted index with non-overflowing informative tokens
        indexed_token_count = 0
        for rec in candidate_records:
            cid = rec["entity_id"]
            # Informative name tokens
            for tok in rec["name_tokens"]:
                if len(tok) >= self.min_token_len and tok not in COMMON_BLOCKING_STOPWORDS:
                    if self.token_counts[tok] <= self.max_token_freq:
                        self.token_index[tok].append(cid)
                        indexed_token_count += 1
            
            # Informative address tokens
            for atok in rec.get("addr_tokens", []):
                if len(atok) >= self.min_token_len and atok not in COMMON_BLOCKING_STOPWORDS:
                    key = f"ADDR:{atok}"
                    if self.token_counts[key] <= self.max_token_freq:
                        self.token_index[key].append(cid)
                        indexed_token_count += 1

            # Postal code index
            if rec.get("postal_code"):
                pc = rec["postal_code"]
                if self.token_counts[pc] <= self.max_token_freq:
                    self.token_index[f"PC:{pc}"].append(cid)

            # Character 3-grams for names
            for ng in rec.get("name_ngrams", []):
                self.ngram_index[ng].append(cid)

        print(f"  Indexed {len(self.token_index):,} unique tokens and {len(self.ngram_index):,} unique n-grams.")

    def generate_candidates(self, query_record: dict, max_candidates_per_query: int = 100) -> List[str]:
        """
        Generate candidate IDs for an S1 query record.
        Combines token hits and n-gram overlap, ranked by frequency weighting.
        """
        candidate_scores = Counter()

        # 1. Token pass: IDF-weighted voting
        query_tokens = query_record.get("name_tokens", set())
        for tok in query_tokens:
            if len(tok) >= self.min_token_len and tok not in COMMON_BLOCKING_STOPWORDS:
                freq = self.token_counts.get(tok, 0)
                if 0 < freq <= self.max_token_freq:
                    idf_weight = 1.0 + math.log(10000.0 / (1.0 + freq))
                    for cid in self.token_index.get(tok, []):
                        candidate_scores[cid] += idf_weight

        # Address token pass (informative street / locality tokens)
        for atok in query_record.get("addr_tokens", set()):
            if len(atok) >= self.min_token_len and atok not in COMMON_BLOCKING_STOPWORDS:
                key = f"ADDR:{atok}"
                freq = self.token_counts.get(key, 0)
                if 0 < freq <= self.max_token_freq:
                    idf_weight = 0.5 + 0.5 * math.log(10000.0 / (1.0 + freq))
                    for cid in self.token_index.get(key, []):
                        candidate_scores[cid] += idf_weight

        # Postal code exact match bonus
        pc = query_record.get("postal_code")
        if pc and f"PC:{pc}" in self.token_index:
            for cid in self.token_index[f"PC:{pc}"]:
                candidate_scores[cid] += 2.0

        # 2. N-gram pass (for queries with few or zero token hits)
        if len(candidate_scores) < 10:
            query_ngrams = query_record.get("name_ngrams", set())
            for ng in query_ngrams:
                for cid in self.ngram_index.get(ng, []):
                    candidate_scores[cid] += 0.25

        if not candidate_scores:
            return []

        # Return top-k candidate IDs sorted by score
        top_candidates = [cid for cid, _ in candidate_scores.most_common(max_candidates_per_query)]
        return top_candidates


def evaluate_blocking_recall(queries: List[dict], ground_truth: Dict[str, Set[str]], blocker: MultiPassBlocker, max_k: int = 100):
    """
    Evaluate candidate recall against known ground truth links.
    Candidate Recall = (Retrieved true matches) / (Total true matches)
    """
    total_true_matches = 0
    retrieved_true_matches = 0
    total_candidates_generated = 0

    for q in queries:
        qid = q["entity_id"]
        true_matches = ground_truth.get(qid, set())
        total_true_matches += len(true_matches)

        candidates = blocker.generate_candidates(q, max_candidates_per_query=max_k)
        total_candidates_generated += len(candidates)
        retrieved_true_matches += len(true_matches.intersection(candidates))

    recall = (retrieved_true_matches / total_true_matches) if total_true_matches > 0 else 1.0
    avg_candidates = total_candidates_generated / len(queries) if queries else 0

    print("\n--- Blocking Evaluation Summary ---")
    print(f"  Total True Matches in Eval: {total_true_matches:,}")
    print(f"  Retrieved True Matches:     {retrieved_true_matches:,}")
    print(f"  Candidate Recall:           {recall * 100:.2f}%")
    print(f"  Avg Candidates per Query:   {avg_candidates:.1f}")

    return {
        "candidate_recall": recall,
        "avg_candidates": avg_candidates,
        "total_true_matches": total_true_matches,
        "retrieved_true_matches": retrieved_true_matches,
    }


def main():
    import os
    import sys
    import argparse
    import pandas as pd

    src_dir = os.path.dirname(os.path.abspath(__file__))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from normalize import preprocess_record

    parser = argparse.ArgumentParser(description="Evaluate multi-pass blocking recall.")
    parser.add_argument("--val-dir", default="data/val", help="Path to validation split")
    parser.add_argument("--data-dir", default="train", help="Path to train directory for candidates")
    parser.add_argument("--n-sample", type=int, default=2000, help="Number of S1 queries to evaluate")
    parser.add_argument("--max-candidates", type=int, default=40, help="Max candidates per query")
    args = parser.parse_args()

    print("==================================================================")
    print("PHASE 3: MULTI-PASS CANDIDATE BLOCKING EVALUATOR")
    print(f"Sample Size: {args.n_sample:,} | Max Candidates: {args.max_candidates}")
    print("==================================================================")

    val_s1_path = os.path.join(args.val_dir, "val_source1.tsv")
    val_gt_path = os.path.join(args.val_dir, "val_ground_truth.tsv")

    if not os.path.exists(val_s1_path) or not os.path.exists(val_gt_path):
        val_s1_path = os.path.join(args.data_dir, "train_source1.tsv")
        val_gt_path = os.path.join(args.data_dir, "train_ground_truth.tsv")

    if not os.path.exists(val_s1_path) or not os.path.exists(val_gt_path):
        print(f"Error: Could not find validation or train datasets.")
        return

    print(f"Loading queries from: {val_s1_path}")
    df_s1 = pd.read_csv(val_s1_path, sep="\t", dtype=str, nrows=args.n_sample)
    df_gt = pd.read_csv(val_gt_path, sep="\t", dtype=str, keep_default_na=False)

    gt_map = {}
    for _, row in df_gt.iterrows():
        s1 = row["source1_entity_id"]
        raw = row["matched_entity_ids"].strip()
        gt_map[s1] = set(m.strip() for m in raw.split(",") if m.strip()) if raw else set()

    queries = []
    needed_positives = set()
    for _, row in df_s1.iterrows():
        rec = preprocess_record(row["entity_id"], row["business_name"], row["business_address"], row.get("country", "US"))
        queries.append(rec)
        needed_positives.update(gt_map.get(row["entity_id"], set()))

    # Build candidate pool
    cand_records = {}
    for fn in ["train_source2.tsv", "train_source3.tsv"]:
        fp = os.path.join(args.data_dir, fn)
        if not os.path.exists(fp):
            continue
        print(f"Streaming candidate records from {fn}...")
        for chunk in pd.read_csv(fp, sep="\t", dtype=str, keep_default_na=False, chunksize=250000):
            matches = chunk[chunk["entity_id"].isin(needed_positives)]
            for _, row in matches.iterrows():
                cand_records[row["entity_id"]] = preprocess_record(
                    row["entity_id"], row["business_name"], row["business_address"], row.get("country", "US")
                )
            if len(cand_records) < len(needed_positives) + 30000:
                sample_chunk = chunk.sample(n=min(len(chunk), 3000), random_state=42)
                for _, row in sample_chunk.iterrows():
                    if row["entity_id"] not in cand_records:
                        cand_records[row["entity_id"]] = preprocess_record(
                            row["entity_id"], row["business_name"], row["business_address"], row.get("country", "US")
                        )

    print(f"Candidate pool ready: {len(cand_records):,} records.")
    blocker = MultiPassBlocker(max_token_freq=6000, min_token_len=3)
    blocker.fit_candidates(list(cand_records.values()))

    evaluate_blocking_recall(queries, gt_map, blocker, max_k=args.max_candidates)
    print("\nPhase 3 blocking evaluation complete.")


if __name__ == "__main__":
    main()
