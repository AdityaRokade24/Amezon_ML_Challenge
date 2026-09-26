#!/usr/bin/env python3
"""
Phase 4: High-Performance Pairwise Feature Engineering Engine
==============================================================
Calculates dense pairwise similarity signals between an S1 entity and candidate records:
- Multi-metric string distances (Levenshtein, Token Sort, Token Set via RapidFuzz)
- Token-level Jaccard & overlap coefficients
- Suffix & structural agreement
- Address, Postal Code, and House Number verification
- Country consistency checks
"""

import numpy as np
from rapidfuzz import fuzz, distance


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute Jaccard index between two sets of tokens."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return float(intersection / union) if union > 0 else 0.0


def overlap_coefficient(set_a: set, set_b: set) -> float:
    """Compute overlap coefficient (intersection / min(|A|, |B|))."""
    if not set_a or not set_b:
        return 0.0
    min_len = min(len(set_a), len(set_b))
    return float(len(set_a.intersection(set_b)) / min_len) if min_len > 0 else 0.0


def extract_pairwise_features(s1_rec: dict, cand_rec: dict) -> dict:
    """
    Extract structured pairwise features for classification.
    Inputs are preprocessed record dictionaries from normalize.py.
    """
    feat = {}

    # ----------------------------------------------------
    # 1. Business Name Similarities
    # ----------------------------------------------------
    name1 = s1_rec.get("name_clean", "")
    name2 = cand_rec.get("name_clean", "")

    # Exact matches
    feat["name_exact_clean"] = 1.0 if (name1 and name1 == name2) else 0.0
    feat["name_exact_base"] = 1.0 if (s1_rec.get("name_base") and s1_rec.get("name_base") == cand_rec.get("name_base")) else 0.0

    # RapidFuzz string metrics (0 to 100 scaled to 0.0 to 1.0)
    if name1 and name2:
        feat["name_levenshtein_ratio"] = fuzz.ratio(name1, name2) / 100.0
        feat["name_token_sort_ratio"] = fuzz.token_sort_ratio(name1, name2) / 100.0
        feat["name_token_set_ratio"] = fuzz.token_set_ratio(name1, name2) / 100.0
        feat["name_partial_ratio"] = fuzz.partial_ratio(name1, name2) / 100.0
    else:
        feat["name_levenshtein_ratio"] = 0.0
        feat["name_token_sort_ratio"] = 0.0
        feat["name_token_set_ratio"] = 0.0
        feat["name_partial_ratio"] = 0.0

    # Token sets
    toks1 = s1_rec.get("name_tokens", set())
    toks2 = cand_rec.get("name_tokens", set())
    feat["name_token_jaccard"] = jaccard_similarity(toks1, toks2)
    feat["name_token_overlap"] = overlap_coefficient(toks1, toks2)

    # 3-gram Jaccard
    ng1 = s1_rec.get("name_ngrams", set())
    ng2 = cand_rec.get("name_ngrams", set())
    feat["name_ngram_jaccard"] = jaccard_similarity(ng1, ng2)

    # Name Length metrics
    len1 = len(name1)
    len2 = len(name2)
    feat["name_len_diff"] = abs(len1 - len2)
    feat["name_len_ratio"] = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0.0

    # Legal Suffix agreement
    suf1 = s1_rec.get("name_suffix", "")
    suf2 = cand_rec.get("name_suffix", "")
    if suf1 and suf2:
        feat["suffix_agreement"] = 1.0 if suf1 == suf2 else -1.0
    else:
        feat["suffix_agreement"] = 0.0

    # ----------------------------------------------------
    # 2. Address Similarities
    # ----------------------------------------------------
    addr1 = s1_rec.get("addr_clean", "")
    addr2 = cand_rec.get("addr_clean", "")

    feat["addr_exact"] = 1.0 if (addr1 and addr1 == addr2) else 0.0

    if addr1 and addr2:
        feat["addr_token_sort_ratio"] = fuzz.token_sort_ratio(addr1, addr2) / 100.0
        feat["addr_token_set_ratio"] = fuzz.token_set_ratio(addr1, addr2) / 100.0
        feat["addr_partial_ratio"] = fuzz.partial_ratio(addr1, addr2) / 100.0
    else:
        feat["addr_token_sort_ratio"] = 0.0
        feat["addr_token_set_ratio"] = 0.0
        feat["addr_partial_ratio"] = 0.0

    atok1 = s1_rec.get("addr_tokens", set())
    atok2 = cand_rec.get("addr_tokens", set())
    feat["addr_token_jaccard"] = jaccard_similarity(atok1, atok2)
    feat["addr_token_overlap"] = overlap_coefficient(atok1, atok2)

    # Postal / PIN code matching
    pc1 = s1_rec.get("postal_code", "")
    pc2 = cand_rec.get("postal_code", "")
    if pc1 and pc2:
        feat["postal_match"] = 1.0 if pc1 == pc2 else -1.0
    else:
        feat["postal_match"] = 0.0

    # Street number matching
    stn1 = s1_rec.get("street_number", "")
    stn2 = cand_rec.get("street_number", "")
    if stn1 and stn2:
        feat["street_num_match"] = 1.0 if stn1 == stn2 else -1.0
    else:
        feat["street_num_match"] = 0.0

    # ----------------------------------------------------
    # 3. Country & Entity Meta Features
    # ----------------------------------------------------
    c1 = str(s1_rec.get("country", "")).strip().upper()
    c2 = str(cand_rec.get("country", "")).strip().upper()
    feat["country_match"] = 1.0 if (c1 and c2 and c1 == c2) else 0.0

    # Source origin flag: 1.0 for S2, 0.0 for S3
    cand_id = cand_rec.get("entity_id", "")
    feat["is_source2"] = 1.0 if cand_id.startswith("S2-") else 0.0

    # ----------------------------------------------------
    # 4. Cross-Field Joint Confirmation Signals
    # ----------------------------------------------------
    # High confidence composite: strong name AND strong address
    feat["name_and_addr_high"] = 1.0 if (feat["name_token_sort_ratio"] > 0.85 and feat["addr_token_sort_ratio"] > 0.70) else 0.0

    return feat


FEATURE_NAMES = [
    "name_exact_clean",
    "name_exact_base",
    "name_levenshtein_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_partial_ratio",
    "name_token_jaccard",
    "name_token_overlap",
    "name_ngram_jaccard",
    "name_len_diff",
    "name_len_ratio",
    "suffix_agreement",
    "addr_exact",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "addr_partial_ratio",
    "addr_token_jaccard",
    "addr_token_overlap",
    "postal_match",
    "street_num_match",
    "country_match",
    "is_source2",
    "name_and_addr_high",
]


def vectorize_features(feature_dicts: list) -> np.ndarray:
    """Convert list of feature dicts to contiguous float32 numpy matrix."""
    matrix = np.empty((len(feature_dicts), len(FEATURE_NAMES)), dtype=np.float32)
    for i, d in enumerate(feature_dicts):
        for j, col in enumerate(FEATURE_NAMES):
            matrix[i, j] = d.get(col, 0.0)
    return matrix


def main():
    import os
    import sys
    src_dir = os.path.dirname(os.path.abspath(__file__))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from normalize import preprocess_record

    print("==================================================================")
    print("PHASE 4: PAIRWISE RAPIDFUZZ FEATURE EXTRACTION ENGINE")
    print(f"Total Dense Pairwise Features: {len(FEATURE_NAMES)}")
    print("==================================================================")

    r1 = preprocess_record("S1-001", "Infosys Technologies Ltd", "Plot 44, Electronic City, Bangalore 560100", "India")
    r2_pos = preprocess_record("S2-001", "Infosys Technologies Private Limited", "Plot No 44, Electronic City, Hosur Rd, Bengaluru, 560100", "India")
    r2_neg = preprocess_record("S2-002", "Tata Consultancy Services Ltd", "Whitefield Main Rd, Bangalore 560066", "India")

    print("\n--- Example 1: TRUE MATCH PAIR ---")
    print(f"  Record A: {r1['raw_name']} | {r1['raw_address']}")
    print(f"  Record B: {r2_pos['raw_name']} | {r2_pos['raw_address']}")
    feat_pos = extract_pairwise_features(r1, r2_pos)
    print("\n  Top Extracted Similarity Features:")
    for k in ["name_levenshtein_ratio", "name_token_sort_ratio", "suffix_agreement", "addr_token_sort_ratio", "postal_match", "name_and_addr_high"]:
        print(f"    - {k:25s}: {feat_pos.get(k):.3f}")

    print("\n--- Example 2: NON-MATCH PAIR ---")
    print(f"  Record A: {r1['raw_name']} | {r1['raw_address']}")
    print(f"  Record B: {r2_neg['raw_name']} | {r2_neg['raw_address']}")
    feat_neg = extract_pairwise_features(r1, r2_neg)
    print("\n  Top Extracted Similarity Features:")
    for k in ["name_levenshtein_ratio", "name_token_sort_ratio", "suffix_agreement", "addr_token_sort_ratio", "postal_match", "name_and_addr_high"]:
        print(f"    - {k:25s}: {feat_neg.get(k):.3f}")

    X_sample = vectorize_features([feat_pos, feat_neg])
    print(f"\nVectorized Feature Matrix Shape: {X_sample.shape} (dtype={X_sample.dtype})")
    print("Phase 4 feature extraction complete.")


if __name__ == "__main__":
    main()
