"""Time-split commodity-return baselines with and without cross-domain features."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

TRADITIONAL_FEATURES = [
    "open", "high", "low", "close", "volume", "daily_return", "return_5d", "return_21d",
    "volatility_5d_ann", "volatility_21d_ann", "volatility_63d_ann", "sma_21", "sma_63",
    "bollinger_z", "rsi_14", "macd", "macd_signal", "momentum_10d", "momentum_21d", "term_spread",
]
MATERIAL_SIGNAL_FEATURES = ["mqi", "mqi_5d_trend", "mqi_21d_trend", "mqi_63d_trend"]
INTERACTION_FEATURES = [
    "supply_disruption_prob", "substitution_elasticity", "green_premium_per_kg",
    "carbon_intensity_virgin", "carbon_intensity_recycled", "herfindahl_index",
]
CROSS_DOMAIN_FEATURES = MATERIAL_SIGNAL_FEATURES + INTERACTION_FEATURES


def prepare_commodity_frame(prices: pd.DataFrame, cross: pd.DataFrame) -> pd.DataFrame:
    """Merge same-day technical and material-financial features, then create next-day target."""
    left, right = prices.copy(), cross.copy()
    left["date"] = pd.to_datetime(left["date"])
    right["date"] = pd.to_datetime(right["date"])
    df = left.merge(right, on=["date", "commodity"], how="left", validate="one_to_one")
    df = df.sort_values(["commodity", "date"]).reset_index(drop=True)
    # Predict the following trading day's log return, never the current day's return.
    df["target_next_return"] = df.groupby("commodity")["daily_return"].shift(-1)
    return df


def _features_with_dummies(df: pd.DataFrame, features: list[str], dummy_columns: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    X = df[features + ["commodity"]].copy()
    X = pd.get_dummies(X, columns=["commodity"], dtype=float)
    if dummy_columns is not None:
        X = X.reindex(columns=dummy_columns, fill_value=0.0)
    return X.astype(float), list(X.columns)


def _score_predictions(actual: np.ndarray, predicted: np.ndarray, dates: pd.Series, group: pd.Series) -> dict[str, Any]:
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[valid], predicted[valid]
    out: dict[str, Any] = {
        "n_test": int(valid.sum()),
        "mae_next_day_log_return": round(float(mean_absolute_error(actual, predicted)), 6),
        "rmse_next_day_log_return": round(float(np.sqrt(mean_squared_error(actual, predicted))), 6),
        "directional_accuracy_pct": round(float(100 * ((actual >= 0) == (predicted >= 0)).mean()), 2),
    }
    # This is a toy directional strategy. Transaction cost is charged when exposure changes.
    df = pd.DataFrame({"date": dates.to_numpy()[valid], "commodity": group.to_numpy()[valid], "actual": actual, "pred": predicted})
    df["position"] = np.sign(df["pred"]).astype(float)
    df = df.sort_values(["commodity", "date"])
    df["turnover"] = df.groupby("commodity")["position"].diff().abs().fillna(df["position"].abs())
    df["strategy_return"] = df["position"] * df["actual"] - 0.001 * df["turnover"]
    df["buy_hold_return"] = df["actual"]
    mean = df["strategy_return"].mean()
    std = df["strategy_return"].std(ddof=1)
    out["strategy_sharpe_annualised"] = round(float(np.sqrt(252) * mean / std), 3) if std and np.isfinite(std) else None
    out["strategy_total_log_return_sum"] = round(float(df["strategy_return"].sum()), 4)
    out["max_drawdown_proxy_pct"] = round(float(100 * (np.exp(df["strategy_return"].cumsum()) / np.exp(df["strategy_return"].cumsum()).cummax() - 1).min()), 2)
    out["transaction_cost_bps_round_trip_assumption"] = 10
    out["warning"] = "Directional toy backtest; not an investment recommendation or fully calibrated production strategy."
    return out


def train_commodity_models(prices: pd.DataFrame, cross: pd.DataFrame, model_dir: Path | str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run chronological ablation test, then fit full-data models for demo inference."""
    df = prepare_commodity_frame(prices, cross)
    features_all = TRADITIONAL_FEATURES + CROSS_DOMAIN_FEATURES
    df = df.dropna(subset=features_all + ["target_next_return", "date"]).copy()
    train_mask = df["date"] < pd.Timestamp("2023-01-01")
    test_mask = ~train_mask
    train, test = df.loc[train_mask].copy(), df.loc[test_mask].copy()
    if train.empty or test.empty:
        raise ValueError("Expected date coverage before and after 2023-01-01 for a time-based holdout.")

    variants: dict[str, list[str]] = {
        "traditional_only": TRADITIONAL_FEATURES,
        "traditional_plus_material": TRADITIONAL_FEATURES + MATERIAL_SIGNAL_FEATURES,
        "traditional_plus_interactions": TRADITIONAL_FEATURES + INTERACTION_FEATURES,
        "traditional_plus_cross_domain": features_all,
    }
    results: dict[str, Any] = {
        "split": "Train dates before 2023-01-01; test dates from 2023-01-01 onward",
        "train_date_min": str(train["date"].min().date()),
        "train_date_max": str(train["date"].max().date()),
        "test_date_min": str(test["date"].min().date()),
        "test_date_max": str(test["date"].max().date()),
        "variants": {},
    }
    models: dict[str, Any] = {}
    feature_cols_by_variant: dict[str, list[str]] = {}
    for variant, features in variants.items():
        X_train, columns = _features_with_dummies(train, features)
        X_test, _ = _features_with_dummies(test, features, columns)
        y_train = train["target_next_return"].to_numpy()
        y_test = test["target_next_return"].to_numpy()
        model = HistGradientBoostingRegressor(
            max_iter=120, max_leaf_nodes=15, learning_rate=0.07,
            l2_regularization=2.0, random_state=42,
        )
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        results["variants"][variant] = _score_predictions(
            y_test, prediction, test["date"].reset_index(drop=True), test["commodity"].reset_index(drop=True)
        )
        models[variant] = model
        feature_cols_by_variant[variant] = columns

    # Fit one full-data model for dashboard demo signals. Performance must be reported from holdout above.
    X_all, latest_columns = _features_with_dummies(df, features_all)
    latest_model = HistGradientBoostingRegressor(
        max_iter=120, max_leaf_nodes=15, learning_rate=0.07,
        l2_regularization=2.0, random_state=42,
    )
    latest_model.fit(X_all, df["target_next_return"].to_numpy())
    models["full_data_for_demo"] = latest_model
    feature_cols_by_variant["full_data_for_demo"] = latest_columns
    if model_dir:
        folder = Path(model_dir)
        folder.mkdir(parents=True, exist_ok=True)
        for name, model in models.items():
            joblib.dump(model, folder / f"commodity_{name}.joblib")
        results["feature_columns"] = feature_cols_by_variant
    return models, results


def predict_latest_commodity_signal(model: Any, prices: pd.DataFrame, cross: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Generate next-day demo signals for the latest row of each commodity."""
    df = prepare_commodity_frame(prices, cross)
    features = TRADITIONAL_FEATURES + CROSS_DOMAIN_FEATURES
    df = df.dropna(subset=features).copy()
    latest = df.sort_values("date").groupby("commodity", as_index=False).tail(1).copy()
    X, _ = _features_with_dummies(latest, features, feature_columns)
    latest["predicted_next_day_log_return"] = model.predict(X)
    latest["predicted_direction"] = np.where(latest["predicted_next_day_log_return"] >= 0, "Up", "Down")
    return latest[["date", "commodity", "close", "mqi", "mqi_21d_trend", "supply_disruption_prob", "substitution_elasticity", "predicted_next_day_log_return", "predicted_direction"]]
