"""
utils.py — Shared preprocessing, feature engineering and scoring helpers.

Extracted from credit_scoring_final.ipynb (Sections 4, 6, 13).
All functions are stateless and safe to call from both train_model.py and app.py.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_home_credit(path: str, sample: int | None = None) -> tuple[pd.DataFrame, str]:
    """
    Load Home Credit Default Risk dataset and apply initial feature transforms.

    Parameters
    ----------
    path   : str          — path to application_train.csv
    sample : int or None  — limit rows (None = full dataset)

    Returns
    -------
    (DataFrame, target_column_name)
    """
    print(f"Loading: {path}")
    df = pd.read_csv(path, nrows=sample)
    print(f"  {len(df):,} rows × {df.shape[1]} columns loaded.")

    # Convert day offsets to interpretable years
    day_mappings = {
        "DAYS_BIRTH":             "AGE_YEARS",
        "DAYS_REGISTRATION":      "REGISTRATION_YEARS",
        "DAYS_ID_PUBLISH":        "ID_PUBLISH_YEARS",
        "DAYS_LAST_PHONE_CHANGE": "PHONE_CHANGE_YEARS",
    }
    for raw_col, new_col in day_mappings.items():
        if raw_col in df.columns:
            df[new_col] = (-df[raw_col] / 365).round(1)

    if "DAYS_EMPLOYED" in df.columns:
        df["DAYS_EMPLOYED"]    = df["DAYS_EMPLOYED"].replace(365243, np.nan)
        df["EMPLOYMENT_YEARS"] = (-df["DAYS_EMPLOYED"] / 365).clip(lower=0).round(1)

    drop_cols = [
        "SK_ID_CURR", "DAYS_BIRTH", "DAYS_EMPLOYED",
        "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    return df, "TARGET"


# ─────────────────────────────────────────────────────────────────────────────
#  Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_dataframe(
    df: pd.DataFrame,
    target_col: str,
    miss_threshold: float = 0.55,
) -> tuple[pd.DataFrame, SimpleImputer, list[str], list[str]]:
    """
    Clean the raw dataframe and return processed data plus fitted artefacts.

    Steps (mirrors notebook Section 6):
      1. Drop columns with > miss_threshold missing values
      2. Remove duplicate rows
      3. Encode categoricals (one-hot ≤10 unique; label-encode >10)
      4. Impute missing numerics with median
      5. Normalise target to strict binary 0/1
      6. Convert booleans to int

    Returns
    -------
    (df_clean, fitted_imputer, low_card_cols, high_card_cols)
    """
    df_clean = df.copy()

    # 1 — Drop high-missing columns
    high_miss = df_clean.isnull().mean()
    high_miss = [c for c in high_miss[high_miss > miss_threshold].index if c != target_col]
    df_clean = df_clean.drop(columns=high_miss)
    print(f"[1/6] Dropped {len(high_miss)} columns exceeding {miss_threshold*100:.0f}% missing.")

    # 2 — Duplicates
    n_before = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    print(f"[2/6] Removed {n_before - len(df_clean):,} duplicate rows.")

    # 3 — Encode categoricals
    cat_cols  = [c for c in df_clean.select_dtypes(include=["object", "category"]).columns
                 if c != target_col]
    low_card  = [c for c in cat_cols if df_clean[c].nunique() <= 10]
    high_card = [c for c in cat_cols if df_clean[c].nunique() > 10]

    le = LabelEncoder()
    for col in high_card:
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))

    if low_card:
        df_clean = pd.get_dummies(df_clean, columns=low_card, drop_first=True)

    print(f"[3/6] Encoded {len(cat_cols)} categorical columns.")

    # 4 — Impute
    num_cols = df_clean.select_dtypes(include=np.number).columns.drop(target_col, errors="ignore")
    imputer  = SimpleImputer(strategy="median")
    df_clean[num_cols] = imputer.fit_transform(df_clean[num_cols])
    print("[4/6] Imputed missing values (median).")

    # 5 — Normalise target
    unique_vals = sorted(df_clean[target_col].dropna().unique())
    if set(unique_vals) == {1, 2}:
        df_clean[target_col] = (df_clean[target_col] == 2).astype(int)
    elif set(unique_vals) <= {0, 1} or set(unique_vals) <= {0.0, 1.0}:
        df_clean[target_col] = df_clean[target_col].astype(int)
    else:
        min_val = min(unique_vals)
        df_clean[target_col] = (df_clean[target_col] != min_val).astype(int)
    print(f"[5/6] Target normalised to binary 0/1.")

    # 6 — Booleans
    bool_cols = df_clean.select_dtypes(include="bool").columns
    df_clean[bool_cols] = df_clean[bool_cols].astype(int)
    print(f"[6/6] Converted {len(bool_cols)} boolean columns.")

    print(f"\nPreprocessing complete — {df_clean.shape[0]:,} rows × {df_clean.shape[1]} cols")
    return df_clean, imputer, low_card, high_card


# ─────────────────────────────────────────────────────────────────────────────
#  Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Add domain-specific features (mirrors notebook Section 6).
    Operates in-place and returns the same dataframe.
    """
    print("Engineering features …")

    if "AMT_CREDIT" in df_clean.columns and "AMT_INCOME_TOTAL" in df_clean.columns:
        df_clean["FEAT_credit_income_ratio"] = (
            df_clean["AMT_CREDIT"] / (df_clean["AMT_INCOME_TOTAL"] + 1)
        ).clip(upper=50).fillna(0)
        print("  + FEAT_credit_income_ratio")

    if "AMT_ANNUITY" in df_clean.columns and "AMT_INCOME_TOTAL" in df_clean.columns:
        df_clean["FEAT_annuity_income_ratio"] = (
            df_clean["AMT_ANNUITY"] / (df_clean["AMT_INCOME_TOTAL"] + 1)
        ).clip(upper=5).fillna(0)
        print("  + FEAT_annuity_income_ratio")

    if "AMT_CREDIT" in df_clean.columns and "AMT_GOODS_PRICE" in df_clean.columns:
        df_clean["FEAT_loan_to_goods"] = (
            df_clean["AMT_CREDIT"] / (df_clean["AMT_GOODS_PRICE"] + 1)
        ).clip(upper=5).fillna(1)
        print("  + FEAT_loan_to_goods")

    if "AGE_YEARS" in df_clean.columns:
        df_clean["FEAT_age_bucket"] = pd.cut(
            df_clean["AGE_YEARS"],
            bins=[0, 25, 35, 50, 65, 120],
            labels=[0, 1, 2, 3, 4],
        ).astype(float).fillna(2)
        print("  + FEAT_age_bucket")

    if "EMPLOYMENT_YEARS" in df_clean.columns and "AGE_YEARS" in df_clean.columns:
        df_clean["FEAT_employment_age_ratio"] = (
            df_clean["EMPLOYMENT_YEARS"] / (df_clean["AGE_YEARS"] + 1)
        ).clip(upper=1).fillna(0)
        print("  + FEAT_employment_age_ratio")

    print(f"Feature engineering complete — {df_clean.shape[1]} total columns.")
    return df_clean


# ─────────────────────────────────────────────────────────────────────────────
#  Credit Scoring Helpers
# ─────────────────────────────────────────────────────────────────────────────

def probability_to_score(prob_default: float, min_score: int = 300, max_score: int = 900) -> int:
    """Map default probability [0, 1] → credit score [300, 900] (CIBIL scale)."""
    return int(np.clip(
        min_score + (max_score - min_score) * (1 - prob_default),
        min_score, max_score,
    ))


def score_to_band(score: int) -> str:
    """Return human-readable risk band for a credit score."""
    if score >= 750:
        return "Excellent"
    if score >= 700:
        return "Good"
    if score >= 650:
        return "Fair"
    if score >= 600:
        return "Poor"
    return "Very Poor"


def get_risk_category(prob_default: float) -> str:
    """
    Simplified risk label for the web UI.
      Low    — prob < 0.20
      Medium — 0.20 ≤ prob < 0.50
      High   — prob ≥ 0.50
    """
    if prob_default < 0.20:
        return "Low"
    if prob_default < 0.50:
        return "Medium"
    return "High"
