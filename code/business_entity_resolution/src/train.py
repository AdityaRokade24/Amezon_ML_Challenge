#!/usr/bin/env python3
"""
Phase 5: Supervised Model Training Engine
=========================================
Trains and compares classifiers on candidate pairs:
1. Logistic Regression (Baseline)
2. Random Forest
3. LightGBM (Primary gradient boosted trees)
4. XGBoost
Evaluates on validation split using exact Macro F_0.5 and optimizes threshold.
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

from features import FEATURE_NAMES, vectorize_features, extract_pairwise_features
from evaluate import compute_macro_f05, optimize_threshold


def train_models(X_train: np.ndarray, y_train: np.ndarray, model_type: str = "lightgbm", **kwargs):
    """Train the chosen classifier on extracted pairwise feature vectors."""
    print(f"\nTraining {model_type} on {len(X_train):,} pairs (Positive class ratio: {y_train.mean():.4f})...")
    
    if model_type == "logistic_regression":
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        model.fit(X_train, y_train)
    elif model_type == "random_forest":
        n_estimators = kwargs.get("n_estimators", 100)
        max_depth = kwargs.get("max_depth", 12)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                       class_weight="balanced", random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
    elif model_type == "xgboost":
        import xgboost as xgb
        pos_weight = (len(y_train) - y_train.sum()) / max(1, y_train.sum())
        model = xgb.XGBClassifier(
            n_estimators=kwargs.get("n_estimators", 200),
            max_depth=kwargs.get("max_depth", 6),
            learning_rate=0.08,
            scale_pos_weight=min(pos_weight, 5.0),
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
    else:  # Default to lightgbm
        pos_weight = (len(y_train) - y_train.sum()) / max(1, y_train.sum())
        model = lgb.LGBMClassifier(
            n_estimators=kwargs.get("n_estimators", 250),
            max_depth=kwargs.get("max_depth", 8),
            learning_rate=0.07,
            num_leaves=31,
            scale_pos_weight=min(pos_weight, 4.0),
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        model.fit(X_train, y_train)

    print(f"  {model_type} training complete.")
    return model


def save_pipeline(model, threshold: float, output_path: str):
    """Save trained model along with optimal threshold and feature metadata."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    bundle = {
        "model": model,
        "threshold": threshold,
        "features": FEATURE_NAMES,
    }
    with open(output_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Pipeline artifact saved to: {output_path}")


def load_pipeline(model_path: str):
    """Load saved model bundle."""
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    return bundle
