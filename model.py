"""
model.py — Prediction-only module.

Loads the saved model bundle (credit_model.pkl) and exposes a single
`predict()` function that accepts an applicant feature dict and returns
the full credit assessment.
"""

import os
import numpy as np
import pandas as pd
import joblib

from utils import probability_to_score, score_to_band, get_risk_category


# ─────────────────────────────────────────────────────────────────────────────
#  Model Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_path: str | None = None) -> dict:
    """
    Load the persisted model bundle.

    Parameters
    ----------
    model_path : str or None
        Path to credit_model.pkl.  Defaults to ``models/credit_model.pkl``
        relative to this file.

    Returns
    -------
    dict with keys: model, scaler, feature_cols, threshold
    """
    if model_path is None:
        base_dir   = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "credit_model.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}.\n"
            "Run  python train_model.py  first to train and save the model."
        )

    bundle = joblib.load(model_path)

    required_keys = {"model", "scaler", "feature_cols", "threshold"}
    missing = required_keys - set(bundle.keys())
    if missing:
        raise ValueError(f"Model bundle is missing keys: {missing}")

    print(f"Model loaded from {model_path}")
    print(f"  Features  : {len(bundle['feature_cols'])}")
    print(f"  Threshold : {bundle['threshold']:.4f}")
    return bundle


# ─────────────────────────────────────────────────────────────────────────────
#  Prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict(applicant_data: dict, bundle: dict) -> dict:
    """
    Run the full inference pipeline for a single applicant.

    Parameters
    ----------
    applicant_data : dict
        Feature-name → value pairs.  Missing features default to 0.
    bundle : dict
        Output of ``load_model()``.

    Returns
    -------
    dict with keys:
        credit_score        (int, 300–900)
        risk_band           (str, Excellent / Good / Fair / Poor / Very Poor)
        risk_category       (str, Low / Medium / High)
        default_probability (float, 0–1 rounded to 4 dp)
        decision            (str, Approve / Decline)
    """
    model        = bundle["model"]
    scaler       = bundle["scaler"]
    feature_cols = bundle["feature_cols"]
    threshold    = bundle["threshold"]

    # Build a single-row DataFrame aligned to the training feature order
    row = {col: applicant_data.get(col, 0) for col in feature_cols}
    X   = pd.DataFrame([row])[feature_cols]

    # Scale using the same scaler used in training
    X_scaled = scaler.transform(X)

    # Ensure correct feature count
    n_feats  = model.n_features_in_ if hasattr(model, "n_features_in_") else len(feature_cols)
    X_scaled = X_scaled[:, :n_feats]

    # Probability of default (class 1)
    prob_default = float(model.predict_proba(X_scaled)[0][1])

    # Credit score and labels
    credit_score  = probability_to_score(prob_default)
    risk_band     = score_to_band(credit_score)
    risk_category = get_risk_category(prob_default)
    decision      = "Approve" if credit_score >= 650 else "Decline"

    return {
        "credit_score":        credit_score,
        "risk_band":           risk_band,
        "risk_category":       risk_category,
        "default_probability": round(prob_default, 4),
        "decision":            decision,
    }
