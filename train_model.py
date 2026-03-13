"""
train_model.py — Training pipeline for the Credit Scoring model.

Run once to produce  models/credit_model.pkl  which is consumed by
model.py and app.py at prediction time.

Usage
-----
    python train_model.py                          # full dataset
    python train_model.py --sample 50000           # first 50 000 rows
    python train_model.py --csv other_file.csv     # custom path
"""

import os
import argparse
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import (
    roc_auc_score, f1_score, precision_score,
    recall_score, accuracy_score, precision_recall_curve,
)
from xgboost import XGBClassifier

from utils import load_home_credit, preprocess_dataframe, engineer_features


# ─────────────────────────────────────────────────────────────────────────────
#  Constants (mirrors notebook Section 2)
# ─────────────────────────────────────────────────────────────────────────────

TEST_SIZE    = 0.2
RANDOM_STATE = 42

XGB_PARAMS = {
    "n_estimators":     1000,
    "max_depth":        6,
    "learning_rate":    0.02,
    "subsample":        0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 5,
    "scale_pos_weight": 11,
    "eval_metric":      "auc",
    "random_state":     RANDOM_STATE,
    "tree_method":      "hist",
    "n_jobs":           -1,
}


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main(csv_path: str, sample: int | None):
    base_dir   = os.path.dirname(os.path.abspath(__file__))
    model_dir  = os.path.join(base_dir, "models")
    model_path = os.path.join(model_dir, "credit_model.pkl")
    os.makedirs(model_dir, exist_ok=True)

    # 1 — Load dataset -------------------------------------------------------
    df, target_col = load_home_credit(csv_path, sample=sample)

    # 2 — Preprocess ---------------------------------------------------------
    df_clean, imputer, low_card, high_card = preprocess_dataframe(df, target_col)

    # 3 — Feature engineering -------------------------------------------------
    df_clean = engineer_features(df_clean)

    # 4 — Prepare X / y ------------------------------------------------------
    feature_cols = [c for c in df_clean.columns if c != target_col]
    X = df_clean[feature_cols].copy()
    y = df_clean[target_col].copy()

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    valid = y.notna()
    X = X[valid].reset_index(drop=True)
    y = y[valid].reset_index(drop=True).astype(int)

    counts = y.value_counts().sort_index()
    print("\nClass distribution:")
    for cls, cnt in counts.items():
        print(f"  Class {cls}: {cnt:,}  ({cnt / len(y) * 100:.2f}%)")

    # 5 — Train / test split --------------------------------------------------
    min_needed   = max(2, int(np.ceil(1 / TEST_SIZE)))
    use_stratify = y if counts.min() >= min_needed else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=use_stratify,
    )

    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    print(f"\nSplit: {X_train.shape[0]:,} train / {X_test.shape[0]:,} test")
    print(f"Features: {X_train.shape[1]}")

    # 6 — Train XGBoost -------------------------------------------------------
    print("\nTraining XGBoost …")
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_train_scaled, y_train)

    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)

    auc_default = roc_auc_score(y_test, y_prob)
    f1_default  = f1_score(y_test, y_pred, zero_division=0)
    print(f"  Default threshold (0.50) — AUC: {auc_default:.4f}  F1: {f1_default:.4f}")

    # 7 — Threshold tuning ----------------------------------------------------
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores   = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx    = f1_scores[:-1].argmax()
    best_thresh = float(thresholds[best_idx])

    y_pred_tuned = (y_prob >= best_thresh).astype(int)
    print(f"\n  Optimal threshold: {best_thresh:.4f}")
    print(f"    AUC       : {roc_auc_score(y_test, y_prob):.4f}")
    print(f"    F1        : {f1_score(y_test, y_pred_tuned):.4f}")
    print(f"    Precision : {precision_score(y_test, y_pred_tuned, zero_division=0):.4f}")
    print(f"    Recall    : {recall_score(y_test, y_pred_tuned):.4f}")
    print(f"    Accuracy  : {accuracy_score(y_test, y_pred_tuned):.4f}")

    # 8 — Collect evaluation metrics -------------------------------------------
    metrics = {
        "auc":       round(roc_auc_score(y_test, y_prob), 4),
        "f1":        round(f1_score(y_test, y_pred_tuned), 4),
        "precision": round(precision_score(y_test, y_pred_tuned, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred_tuned), 4),
        "accuracy":  round(accuracy_score(y_test, y_pred_tuned), 4),
    }

    # 9 — Feature importances (top 20) ----------------------------------------
    importances = model.feature_importances_
    n_feats_model = min(len(importances), len(feature_cols))
    fi_series = pd.Series(importances[:n_feats_model], index=feature_cols[:n_feats_model])
    fi_top = fi_series.sort_values(ascending=False).head(20)
    feature_importances = {name: round(float(val), 6) for name, val in fi_top.items()}

    # 10 — Save model bundle ---------------------------------------------------
    bundle = {
        "model":               model,
        "scaler":              scaler,
        "feature_cols":        feature_cols,
        "threshold":           best_thresh,
        "metrics":             metrics,
        "feature_importances": feature_importances,
    }
    joblib.dump(bundle, model_path)
    size_mb = os.path.getsize(model_path) / 1e6
    print(f"\nModel bundle saved → {model_path}  ({size_mb:.1f} MB)")
    print("Keys:", list(bundle.keys()))


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Credit Scoring Model")
    parser.add_argument(
        "--csv", default="application_train.csv",
        help="Path to application_train.csv (default: application_train.csv)",
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Limit rows loaded (default: full dataset)",
    )
    args = parser.parse_args()
    main(csv_path=args.csv, sample=args.sample)
