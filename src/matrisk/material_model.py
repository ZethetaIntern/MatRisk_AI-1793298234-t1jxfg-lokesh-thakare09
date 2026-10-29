"""Material property baseline with chemical-system-aware holdout evaluation.

This module intentionally implements an ExtraTrees baseline, not a CGNN. It uses
composition strings plus structural descriptors. Crystal graph construction and a trained
physics-informed neural network are future extensions, not claims of this prototype.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGETS = [
    "formation_energy_per_atom_eV",
    "band_gap_eV",
    "bulk_modulus_GPa",
    "shear_modulus_GPa",
]
NUMERIC = ["n_elements", "spacegroup_number", "nsites", "volume_A3"]
CATEGORICAL = ["crystal_system", "category"]
FEATURES = ["formula", *NUMERIC, *CATEGORICAL]
TARGET_LABELS = {
    "formation_energy_per_atom_eV": "Formation energy (eV/atom)",
    "band_gap_eV": "Band gap (eV)",
    "bulk_modulus_GPa": "Bulk modulus (GPa)",
    "shear_modulus_GPa": "Shear modulus (GPa)",
}


def chemical_system(formula: str) -> str:
    """Group formulas by elemental system, e.g. Fe2O3 and FeO -> Fe-O."""
    elements = sorted(set(re.findall(r"[A-Z][a-z]?", str(formula))))
    return "-".join(elements) if elements else "UNKNOWN"


def make_material_pipeline(random_state: int = 42) -> Pipeline:
    """Create an interpretable-enough tree baseline over composition and structure features."""
    prep = ColumnTransformer(
        transformers=[
            ("formula_chargrams", TfidfVectorizer(analyzer="char", ngram_range=(1, 2), min_df=2, max_features=800), "formula"),
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), NUMERIC),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), CATEGORICAL),
        ],
        remainder="drop",
    )
    regressor = ExtraTreesRegressor(
        n_estimators=80,
        max_depth=24,
        min_samples_leaf=3,
        max_features=0.20,
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline([("features", prep), ("model", regressor)])


def material_quality_checks(frame: pd.DataFrame) -> dict[str, Any]:
    """Audit basic bounds and elasticity consistency in the provided material records."""
    k = pd.to_numeric(frame["bulk_modulus_GPa"], errors="coerce")
    g = pd.to_numeric(frame["shear_modulus_GPa"], errors="coerce")
    nu = pd.to_numeric(frame["poisson_ratio"], errors="coerce")
    density = pd.to_numeric(frame["density_g_cm3"], errors="coerce")
    lhs = 2 * g * (1 + nu)
    rhs = 3 * k * (1 - 2 * nu)
    denom = ((lhs.abs() + rhs.abs()) / 2).replace(0, np.nan)
    relative_gap = (lhs - rhs).abs() / denom
    valid_bounds = (k > 0) & (g > 0) & (density > 0) & (nu > -1) & (nu < 0.5)
    elastically_consistent = relative_gap <= 0.10
    return {
        "rows": int(len(frame)),
        "positive_K_G_density_pct": round(100 * ((k > 0) & (g > 0) & (density > 0)).mean(), 2),
        "poisson_bounds_pct": round(100 * ((nu > -1) & (nu < 0.5)).mean(), 2),
        "all_basic_bounds_pct": round(100 * valid_bounds.mean(), 2),
        "elastic_consistency_10pct_pct": round(100 * elastically_consistent.fillna(False).mean(), 2),
        "median_elastic_relative_gap_pct": round(100 * relative_gap.median(), 2),
        "note": "Elastic consistency uses a 10% relative-gap tolerance; this is a data audit, not a PINN result.",
    }


def train_material_model(frame: pd.DataFrame, model_path: Path | str | None = None) -> tuple[Pipeline, dict[str, Any]]:
    """Fit an ExtraTrees baseline using a holdout split grouped by elemental system."""
    df = frame.dropna(subset=FEATURES + TARGETS).copy()
    df["_chemical_system"] = df["formula"].map(chemical_system)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(splitter.split(df[FEATURES], df[TARGETS], groups=df["_chemical_system"]))
    X_train, X_test = df.iloc[train_idx][FEATURES], df.iloc[test_idx][FEATURES]
    y_train, y_test = df.iloc[train_idx][TARGETS], df.iloc[test_idx][TARGETS]
    model = make_material_pipeline()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "split": "GroupShuffleSplit by elemental system (20% holdout)",
        "per_target": {},
        "physics_audit_raw_data": material_quality_checks(frame),
    }
    for i, target in enumerate(TARGETS):
        metrics["per_target"][target] = {
            "label": TARGET_LABELS[target],
            "mae": round(float(mean_absolute_error(y_test[target], pred[:, i])), 5),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test[target], pred[:, i]))), 5),
            "r2": round(float(r2_score(y_test[target], pred[:, i])), 4),
        }
    if model_path:
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
    return model, metrics


def predict_material(model: Pipeline, record: pd.Series | dict[str, Any]) -> dict[str, float]:
    """Predict four properties and derive a physically consistent Poisson ratio from K and G."""
    row = dict(record)
    X = pd.DataFrame([{feature: row.get(feature) for feature in FEATURES}])
    raw = model.predict(X)[0]
    result = {target: float(value) for target, value in zip(TARGETS, raw)}
    k = max(result["bulk_modulus_GPa"], 1e-6)
    g = max(result["shear_modulus_GPa"], 1e-6)
    nu = (3 * k - 2 * g) / (2 * (3 * k + g))
    # Conservative clipping avoids a numerical edge case in a poorly supported region.
    nu = float(np.clip(nu, -0.99, 0.499))
    young_from_shear = 2 * g * (1 + nu)
    young_from_bulk = 3 * k * (1 - 2 * nu)
    result.update({
        "poisson_ratio_derived": nu,
        "young_modulus_from_shear_GPa": float(young_from_shear),
        "young_modulus_from_bulk_GPa": float(young_from_bulk),
        "elastic_relative_gap_pct": float(100 * abs(young_from_shear - young_from_bulk) / max((young_from_shear + young_from_bulk) / 2, 1e-9)),
        "physics_bounds_valid": bool(k > 0 and g > 0 and -1 < nu < 0.5),
    })
    return result
