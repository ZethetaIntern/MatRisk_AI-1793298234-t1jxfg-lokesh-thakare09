"""Bridge condition risk classifier and transparent financial-loss arithmetic."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BRIDGE_NUMERIC = [
    "age_years", "design_life_years", "corrosion_rate_mm_yr", "adt", "deck_area_sqft",
    "tensile_strength_MPa", "yield_strength_MPa", "original_thickness_mm",
    "remaining_thickness_mm", "fatigue_cycles_millions",
]
BRIDGE_CATEGORICAL = ["bridge_type", "material", "location", "corrosion_environment"]
BRIDGE_FEATURES = BRIDGE_NUMERIC + BRIDGE_CATEGORICAL


def make_bridge_pipeline() -> Pipeline:
    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), BRIDGE_NUMERIC),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), BRIDGE_CATEGORICAL),
    ])
    clf = RandomForestClassifier(
        n_estimators=180, min_samples_leaf=4, class_weight="balanced_subsample",
        max_features="sqrt", random_state=42, n_jobs=-1,
    )
    return Pipeline([("features", prep), ("model", clf)])


def train_bridge_model(bridges: pd.DataFrame, model_path: Path | str | None = None) -> tuple[Pipeline, dict[str, Any]]:
    df = bridges.dropna(subset=BRIDGE_FEATURES + ["structurally_deficient"]).copy()
    X = df[BRIDGE_FEATURES]
    y = df["structurally_deficient"].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = make_bridge_pipeline()
    model.fit(x_train, y_train)
    p = model.predict_proba(x_test)[:, 1]
    pred = (p >= 0.5).astype(int)
    metrics = {
        "n_train": int(len(x_train)), "n_test": int(len(x_test)),
        "positive_prevalence_pct_test": round(100 * float(y_test.mean()), 2),
        "roc_auc": round(float(roc_auc_score(y_test, p)), 4) if y_test.nunique() > 1 else None,
        "average_precision": round(float(average_precision_score(y_test, p)), 4),
        "brier_score": round(float(brier_score_loss(y_test, p)), 5),
        "classification_report_at_0_5": classification_report(y_test, pred, output_dict=True, zero_division=0),
        "target_definition": "structurally_deficient (0/1) in the supplied sample file",
        "warning": "Classifier probability is a deficiency-risk proxy, not a calibrated annual loan-default probability.",
    }
    if model_path:
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
    return model, metrics


def financial_risk_metrics(
    bridges: pd.DataFrame,
    model: Pipeline,
    failures: pd.DataFrame,
    max_assets: int | None = None,
) -> pd.DataFrame:
    """Calculate PD-proxy x LGD x EAD using sample historical-loss ratios."""
    df = bridges.copy()
    if max_assets is not None:
        df = df.head(max_assets).copy()
    df["pd_proxy"] = model.predict_proba(df[BRIDGE_FEATURES])[:, 1]
    loss_ratio = pd.to_numeric(failures["loss_ratio"], errors="coerce").dropna().clip(0, 1)
    lgd_base = float(loss_ratio.mean()) if not loss_ratio.empty else 0.30
    # Slight severity adjustment from current observed condition; explicitly illustrative.
    condition = pd.to_numeric(df.get("condition_rating", 7), errors="coerce").fillna(7)
    df["lgd_proxy"] = np.clip(lgd_base + (7 - condition) * 0.025, 0.05, 0.95)
    df["ead_M"] = pd.to_numeric(df["loan_outstanding_M"], errors="coerce").fillna(0).clip(lower=0)
    df["expected_loss_proxy_M"] = df["pd_proxy"] * df["lgd_proxy"] * df["ead_M"]
    df["risk_band"] = pd.cut(df["pd_proxy"], bins=[-0.001, 0.05, 0.15, 1.001], labels=["Lower", "Moderate", "Higher"])
    return df
