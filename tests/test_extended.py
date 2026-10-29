from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from matrisk.data import DATA_DIR, FILES, dataset_summary, load_dataset
from matrisk.material_model import TARGETS, chemical_system, material_quality_checks, train_material_model, predict_material
from matrisk.commodity_model import (
    TRADITIONAL_FEATURES, CROSS_DOMAIN_FEATURES, train_commodity_models,
    predict_latest_commodity_signal, _score_predictions, _features_with_dummies,
)
from matrisk.bridge_risk import BRIDGE_FEATURES, train_bridge_model, financial_risk_metrics
from matrisk.quality import quality_report
from matrisk.inverse_design import elemental_cost_proxy, screen_existing_materials
from matrisk.esg import carbon_pair_for_commodity, substitution_impact


def test_data_loader_unknown_missing_and_summary(tmp_path):
    with pytest.raises(KeyError):
        load_dataset("not-a-dataset")
    with pytest.raises(FileNotFoundError):
        load_dataset("materials", tmp_path)
    summary = dataset_summary(DATA_DIR)
    assert set(summary["dataset"]) == set(FILES)
    assert (summary["rows"] > 0).all()


def test_material_audit_handles_physical_bounds():
    df = pd.DataFrame({
        "bulk_modulus_GPa": [100, -1], "shear_modulus_GPa": [40, 0],
        "poisson_ratio": [0.3, 0.8], "density_g_cm3": [2.5, -1],
    })
    result = material_quality_checks(df)
    assert result["rows"] == 2
    assert result["all_basic_bounds_pct"] == 50.0
    assert result["poisson_bounds_pct"] == 50.0


def test_material_training_and_saved_prediction(tmp_path):
    frame = pd.read_csv(DATA_DIR / FILES["materials"]).sample(360, random_state=20)
    model_path = tmp_path / "models" / "material.joblib"
    model, metrics = train_material_model(frame, model_path)
    assert model_path.exists()
    assert metrics["n_train"] > metrics["n_test"]
    assert set(metrics["per_target"]) == set(TARGETS)
    row = frame.iloc[0].to_dict()
    prediction = predict_material(model, row)
    assert prediction["physics_bounds_valid"]
    assert prediction["elastic_relative_gap_pct"] < 1e-5


def _toy_commodity_frames():
    rng = np.random.default_rng(24)
    commodities = ["Steel_HRC", "Aluminium_LME", "Copper_COMEX", "Nickel_LME"]
    dates = list(pd.bdate_range("2022-10-01", periods=45)) + list(pd.bdate_range("2023-01-01", periods=45))
    price_rows, cross_rows = [], []
    for commodity in commodities:
        price = 900.0
        for date in dates:
            daily_return = float(rng.normal(0, 0.015))
            price *= float(np.exp(daily_return))
            row = {"date": date, "commodity": commodity, "daily_return": daily_return,
                   "open": price * 0.999, "high": price * 1.01, "low": price * 0.99,
                   "close": price, "volume": float(rng.integers(100, 10000)),
                   "return_5d": rng.normal(0, .02), "return_21d": rng.normal(0, .04),
                   "volatility_5d_ann": 25 + rng.random() * 8,
                   "volatility_21d_ann": 25 + rng.random() * 8,
                   "volatility_63d_ann": 25 + rng.random() * 8,
                   "sma_21": price * 0.99, "sma_63": price * 0.98,
                   "bollinger_z": rng.normal(), "rsi_14": rng.uniform(20,80),
                   "macd": rng.normal(), "macd_signal": rng.normal(),
                   "momentum_10d": rng.normal(), "momentum_21d": rng.normal(),
                   "term_spread": rng.normal(0, 5)}
            price_rows.append(row)
            cross_rows.append({"date": date, "commodity": commodity, "mqi": rng.uniform(50, 90),
                               "mqi_5d_trend": rng.normal(), "mqi_21d_trend": rng.normal(),
                               "mqi_63d_trend": rng.normal(), "supply_disruption_prob": rng.uniform(0,1),
                               "substitution_elasticity": rng.uniform(0,1), "green_premium_per_kg": rng.uniform(0,2),
                               "carbon_intensity_virgin": rng.uniform(.4,12), "carbon_intensity_recycled": rng.uniform(.1,1),
                               "herfindahl_index": rng.uniform(0,1)})
    return pd.DataFrame(price_rows), pd.DataFrame(cross_rows)


def test_commodity_feature_matrices_score_models_and_latest_signal(tmp_path):
    prices, cross = _toy_commodity_frames()
    X, cols = _features_with_dummies(prices, TRADITIONAL_FEATURES)
    X2, _ = _features_with_dummies(prices, TRADITIONAL_FEATURES, cols)
    assert X.shape == X2.shape and np.isfinite(X.to_numpy()).all()
    scored = _score_predictions(np.array([.01, -.02, np.nan]), np.array([.02, -.01, .3]),
                                pd.Series(pd.date_range("2024-01-01", periods=3)),
                                pd.Series(["Steel_HRC"]*3))
    assert scored["n_test"] == 2
    models, metrics = train_commodity_models(prices, cross, tmp_path / "models")
    assert set(metrics["variants"]) == {
        "traditional_only", "traditional_plus_material", "traditional_plus_interactions", "traditional_plus_cross_domain"
    }
    assert (tmp_path / "models" / "commodity_full_data_for_demo.joblib").exists()
    latest = predict_latest_commodity_signal(models["full_data_for_demo"], prices, cross, metrics["feature_columns"]["full_data_for_demo"])
    assert latest["commodity"].nunique() == prices["commodity"].nunique()
    assert set(latest["predicted_direction"]) <= {"Up", "Down"}


def test_bridge_model_and_financial_proxy(tmp_path):
    bridges = pd.read_csv(DATA_DIR / FILES["bridges"]).sample(1200, random_state=44)
    failures = pd.read_csv(DATA_DIR / FILES["failures"])
    model_path = tmp_path / "bridge.joblib"
    model, metrics = train_bridge_model(bridges, model_path)
    assert model_path.exists()
    assert 0 <= metrics["roc_auc"] <= 1
    results = financial_risk_metrics(bridges.head(25), model, failures, max_assets=12)
    assert len(results) == 12
    assert (results["expected_loss_proxy_M"] >= 0).all()
    assert set(results["risk_band"].dropna().astype(str)) <= {"Lower", "Moderate", "Higher"}


def test_quality_report_and_inverse_material_screening():
    materials = pd.read_csv(DATA_DIR / FILES["materials"])
    prices = pd.read_csv(DATA_DIR / FILES["prices"])
    cross = pd.read_csv(DATA_DIR / FILES["cross_domain"])
    bridges = pd.read_csv(DATA_DIR / FILES["bridges"])
    failures = pd.read_csv(DATA_DIR / FILES["failures"])
    datasets = {"materials": materials, "prices": prices, "cross_domain": cross, "bridges": bridges, "failures": failures}
    report = quality_report(datasets)
    assert report["checks"]["commodity_count"] == 8
    assert report["checks"]["cross_domain_key_duplicates"] == 0
    assert report["datasets"]["materials"]["rows"] == 5500
    assert elemental_cost_proxy("Fe", pd.read_csv(DATA_DIR / FILES["element_prices"])) is not None
    candidates = screen_existing_materials(materials.head(120), pd.read_csv(DATA_DIR / FILES["element_prices"]),
                                          target_bulk_GPa=100, max_density_g_cm3=8, max_cost_proxy_usd_kg=100, top_n=5)
    assert len(candidates) == 5
    assert "screen_score" in candidates


def test_esg_pair_and_zero_avoidance():
    cross = pd.read_csv(DATA_DIR / FILES["cross_domain"])
    pair = carbon_pair_for_commodity(cross, "Steel_HRC")
    assert pair["carbon_virgin"] > pair["carbon_recycled"]
    no_savings = substitution_impact(100, 1, 1, .2)
    assert no_savings["cost_per_tco2e_avoided_usd"] is None
    with pytest.raises(KeyError):
        carbon_pair_for_commodity(cross, "made-up commodity")


def test_train_all_main_writes_metrics_with_isolated_output(monkeypatch, tmp_path):
    import matrisk.train_all as runner
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(runner, "OUTPUT_DIR", tmp_path / "outputs")
    monkeypatch.setattr(runner, "train_material_model", lambda frame, path: (None, {"model": "material-test", "per_target": {"toy": {"mae": 0.1}}}))
    monkeypatch.setattr(runner, "train_commodity_models", lambda prices, cross, path: ({}, {"variants": {"toy": {"mae": 0.01}}}))
    monkeypatch.setattr(runner, "train_bridge_model", lambda bridges, path: (None, {"roc_auc": 0.5, "average_precision": 0.1, "brier_score": 0.2}))
    monkeypatch.setattr(runner, "load_all", lambda: {
        "materials": pd.DataFrame({"bulk_modulus_GPa": [10], "shear_modulus_GPa": [5], "poisson_ratio": [.2], "density_g_cm3": [2]}),
        "prices": pd.DataFrame({"commodity": ["Steel_HRC"], "date": ["2024-01-01"]}),
        "bridges": pd.DataFrame({"structurally_deficient": [0]}),
        "cross_domain": pd.DataFrame({"date": ["2024-01-01"], "commodity": ["Steel_HRC"]}),
        "element_prices": pd.DataFrame({"element": ["Fe"]}),
        "failures": pd.DataFrame({"loss_ratio": [.2]}),
    })
    metrics = runner.main()
    assert metrics["material_model"]["model"] == "material-test"
    assert (tmp_path / "outputs" / "metrics.json").exists()
    assert (tmp_path / "outputs" / "dataset_summary.csv").exists()


def test_api_functions_valid_and_not_found():
    from matrisk.api import health, material, stress, StressRequest
    import fastapi
    assert health()["status"] == "ok"
    out = stress(StressRequest(scenario="Climate corrosion", exposure_usd=100000, n_paths=100))
    assert out["n_paths"] == 100 and "samples" not in out
    with pytest.raises(fastapi.HTTPException) as exc:
        material("not-a-real-material")
    assert exc.value.status_code == 404
    sample_id = pd.read_csv(DATA_DIR / FILES["materials"], usecols=["material_id"]).iloc[0,0]
    valid = material(str(sample_id))
    assert valid["material_id"] == sample_id
    assert valid["predictions"]["physics_bounds_valid"]
