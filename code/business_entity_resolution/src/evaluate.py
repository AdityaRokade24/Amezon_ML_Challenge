#!/usr/bin/env python3
"""
Phase 6: Official Macro F_0.5 Metric & High-Precision Threshold Optimizer
==========================================================================
Computes the exact competition evaluation metric:
- Macro-averaged F_0.5 across all Source 1 entities
- Full credit (1.0) for correctly identified singletons (predicted empty == true empty)
- Heavy penalty (0.0) for false merges on singletons
- Precision-weighted (beta = 0.5) scoring for entities with true matches
- Optimal decision threshold grid search
"""

import numpy as np
from typing import Dict, Set, List, Tuple


def compute_entity_f05(pred_matches: Set[str], true_matches: Set[str]) -> float:
    """
    Compute F_0.5 for a single S1 entity according to official competition rules:
    - Singletons: 1.0 if pred is empty, 0.0 if pred is non-empty.
    - True entities: 0.0 if pred is empty.
    - Non-empty overlap: F_0.5 = (1.25 * P * R) / (0.25 * P + R)
    """
    n_true = len(true_matches)
    n_pred = len(pred_matches)

    # Singleton case
    if n_true == 0:
        return 1.0 if n_pred == 0 else 0.0

    # Non-singleton but model predicted no matches
    if n_pred == 0:
        return 0.0

    # General case
    tp = len(pred_matches.intersection(true_matches))
    if tp == 0:
        return 0.0

    precision = tp / n_pred
    recall = tp / n_true

    denom = 0.25 * precision + recall
    if denom == 0.0:
        return 0.0
    return (1.25 * precision * recall) / denom


def compute_macro_f05(predictions: Dict[str, Set[str]], ground_truth: Dict[str, Set[str]]) -> Tuple[float, dict]:
    """
    Compute macro-average F_0.5 across all S1 entities in ground_truth.
    Returns:
        macro_f05: float
        details: dict of breakdown stats (singleton accuracy, precision, recall)
    """
    all_s1_ids = list(ground_truth.keys())
    scores = []
    singleton_correct = 0
    singleton_total = 0
    non_singleton_scores = []

    for s1_id in all_s1_ids:
        true_set = ground_truth.get(s1_id, set())
        pred_set = predictions.get(s1_id, set())
        
        score = compute_entity_f05(pred_set, true_set)
        scores.append(score)

        if len(true_set) == 0:
            singleton_total += 1
            if len(pred_set) == 0:
                singleton_correct += 1
        else:
            non_singleton_scores.append(score)

    macro_f05 = float(np.mean(scores)) if scores else 0.0
    singleton_acc = (singleton_correct / singleton_total) if singleton_total > 0 else 1.0
    non_singleton_f05 = float(np.mean(non_singleton_scores)) if non_singleton_scores else 0.0

    details = {
        "macro_f05": macro_f05,
        "total_s1": len(all_s1_ids),
        "singleton_total": singleton_total,
        "singleton_accuracy": singleton_acc,
        "non_singleton_entities": len(non_singleton_scores),
        "non_singleton_f05": non_singleton_f05,
    }
    return macro_f05, details


def optimize_threshold(candidate_scores: List[Tuple[str, str, float]], 
                       ground_truth: Dict[str, Set[str]], 
                       thresholds: List[float] = None) -> Tuple[float, float, dict]:
    """
    Perform grid search to find the probability threshold that maximizes Macro F_0.5.
    candidate_scores: list of (s1_id, cand_id, probability)
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.50, 0.96, 0.02)]

    best_thresh = 0.50
    best_f05 = -1.0
    best_details = {}

    print(f"Optimizing decision threshold across {len(thresholds)} values from {thresholds[0]} to {thresholds[-1]}...")

    # Organize candidates by s1_id for fast filtering
    s1_to_cands = {}
    for s1_id, cand_id, prob in candidate_scores:
        if s1_id not in s1_to_cands:
            s1_to_cands[s1_id] = []
        s1_to_cands[s1_id].append((cand_id, prob))

    all_s1_ids = list(ground_truth.keys())

    for thresh in thresholds:
        preds = {}
        for s1_id in all_s1_ids:
            cand_list = s1_to_cands.get(s1_id, [])
            matched = {cid for cid, p in cand_list if p >= thresh}
            preds[s1_id] = matched

        f05, details = compute_macro_f05(preds, ground_truth)
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh
            best_details = details

    print(f"\nOptimization Result:")
    print(f"  Best Threshold:       {best_thresh:.2f}")
    print(f"  Best Macro F_0.5:     {best_f05:.4f}")
    print(f"  Singleton Accuracy:   {best_details['singleton_accuracy'] * 100:.2f}%")
    print(f"  Non-singleton F_0.5:  {best_details['non_singleton_f05']:.4f}")

    return best_thresh, best_f05, best_details
