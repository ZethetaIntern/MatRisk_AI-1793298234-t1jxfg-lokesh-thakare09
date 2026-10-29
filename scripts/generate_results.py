"""Regenerate result plots from the local sample datasets and trained models."""
from __future__ import annotations
import json, sys
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.model_selection import GroupShuffleSplit, train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from matrisk.data import load_all
from matrisk.material_model import FEATURES, TARGETS, TARGET_LABELS, chemical_system
from matrisk.commodity_model import TRADITIONAL_FEATURES, CROSS_DOMAIN_FEATURES, prepare_commodity_frame, _features_with_dummies
from matrisk.bridge_risk import BRIDGE_FEATURES
from matrisk.stress import SCENARIOS, run_stress_test
from matrisk.esg import carbon_pair_for_commodity, substitution_impact

plt.rcParams.update({"figure.figsize": (8, 5), "figure.dpi": 140, "axes.grid": True})
DATA = load_all()
FIG = ROOT / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
metrics = json.loads((ROOT / "outputs" / "metrics.json").read_text(encoding="utf-8"))

# 1. Material property holdout performance by grouped elemental system.
mat = DATA["materials"].dropna(subset=FEATURES + TARGETS).copy()
mat["_system"] = mat["formula"].map(chemical_system)
tr, te = next(GroupShuffleSplit(n_splits=1, test_size=.20, random_state=42).split(mat[FEATURES], mat[TARGETS], groups=mat["_system"]))
model = joblib.load(ROOT / "models" / "material_extratrees.joblib")
pred = model.predict(mat.iloc[te][FEATURES])
for i, target in enumerate(TARGETS):
    fig, ax = plt.subplots()
    y = mat.iloc[te][target].to_numpy()
    ax.scatter(y, pred[:, i], alpha=.35, s=12)
    lo = min(float(np.nanmin(y)), float(np.nanmin(pred[:, i])))
    hi = max(float(np.nanmax(y)), float(np.nanmax(pred[:, i])))
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlabel("Observed sample value")
    ax.set_ylabel("Model prediction")
    ax.set_title(f"Material holdout: {TARGET_LABELS[target]}")
    fig.tight_layout(); fig.savefig(FIG / f"material_{target}.png"); plt.close(fig)

# 2. Commodity ablation summary.
variants = metrics["commodity_model"]["variants"]
fig, ax = plt.subplots(figsize=(10, 5))
labels = ["A: Technical", "B: + MQI signals", "C: + interaction", "D: Full"]
keys = ["traditional_only", "traditional_plus_material", "traditional_plus_interactions", "traditional_plus_cross_domain"]
metric_vals = [variants[k]["directional_accuracy_pct"] for k in keys]
bars = ax.bar(labels, metric_vals)
ax.axhline(50, linestyle="--", linewidth=1, label="50% directional reference")
ax.set_ylim(45, 55)
ax.set_ylabel("Directional accuracy (%)")
ax.set_title("Four-way commodity ablation on 2023-2024 holdout")
ax.legend()
for bar, val in zip(bars, metric_vals): ax.text(bar.get_x()+bar.get_width()/2, val+.1, f"{val:.2f}%", ha="center")
fig.tight_layout(); fig.savefig(FIG / "commodity_ablation_direction.png"); plt.close(fig)

# 3. Commodity sample strategy equity curve, calculated on holdout only.
df = prepare_commodity_frame(DATA["prices"], DATA["cross_domain"])
features_all = TRADITIONAL_FEATURES + CROSS_DOMAIN_FEATURES
df = df.dropna(subset=features_all + ["target_next_return", "date"]).copy()
test = df.loc[df["date"] >= pd.Timestamp("2023-01-01")].copy()
model_names = ["traditional_only", "traditional_plus_cross_domain"]
fig, ax = plt.subplots(figsize=(10, 5))
from matrisk.commodity_model import MATERIAL_SIGNAL_FEATURES, INTERACTION_FEATURES
for variant, label, features in [
    ("traditional_only", "A: Technical", TRADITIONAL_FEATURES),
    ("traditional_plus_material", "B: + MQI", TRADITIONAL_FEATURES + MATERIAL_SIGNAL_FEATURES),
    ("traditional_plus_interactions", "C: + interactions", TRADITIONAL_FEATURES + INTERACTION_FEATURES),
    ("traditional_plus_cross_domain", "D: Full", features_all),
]:
    model = joblib.load(ROOT / "models" / f"commodity_{variant}.joblib")
    X, columns = _features_with_dummies(df.loc[df["date"] < pd.Timestamp("2023-01-01")], features)
    X_test, _ = _features_with_dummies(test, features, columns)
    pred_ret = model.predict(X_test)
    bt = test[["date", "commodity", "target_next_return"]].copy()
    bt["pred"] = pred_ret
    bt["position"] = np.sign(bt["pred"])
    bt = bt.sort_values(["commodity", "date"])
    bt["turnover"] = bt.groupby("commodity")["position"].diff().abs().fillna(bt["position"].abs())
    bt["strategy_return"] = bt["position"] * bt["target_next_return"] - .001 * bt["turnover"]
    portfolio = bt.groupby("date")["strategy_return"].mean().sort_index()
    ax.plot(portfolio.index, np.exp(portfolio.cumsum()), label=label)
ax.set_title("Toy strategy equity index on chronological holdout")
ax.set_xlabel("Date"); ax.set_ylabel("Equity index (start = 1)"); ax.legend()
fig.tight_layout(); fig.savefig(FIG / "commodity_equity_curve.png"); plt.close(fig)

# 4. Bridge classifier holdout ROC / precision-recall.
bridges = DATA["bridges"].dropna(subset=BRIDGE_FEATURES + ["structurally_deficient"]).copy()
X = bridges[BRIDGE_FEATURES]; y = bridges["structurally_deficient"].astype(int)
xt, xv, yt, yv = train_test_split(X, y, test_size=.20, random_state=42, stratify=y)
bmodel = joblib.load(ROOT / "models" / "bridge_deficiency_rf.joblib")
proba = bmodel.predict_proba(xv)[:,1]
fpr, tpr, _ = roc_curve(yv, proba)
precision, recall, _ = precision_recall_curve(yv, proba)
fig, ax = plt.subplots(); ax.plot(fpr,tpr,label=f"ROC AUC={metrics['bridge_model']['roc_auc']:.3f}"); ax.plot([0,1],[0,1],linestyle="--"); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.set_title("Bridge deficiency classifier — holdout ROC"); ax.legend(); fig.tight_layout(); fig.savefig(FIG / "bridge_roc.png"); plt.close(fig)
fig, ax = plt.subplots(); ax.plot(recall,precision,label=f"Average precision={metrics['bridge_model']['average_precision']:.3f}"); ax.axhline(yv.mean(), linestyle="--", label=f"Prevalence={yv.mean():.3f}"); ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Bridge deficiency classifier — holdout precision/recall"); ax.legend(); fig.tight_layout(); fig.savefig(FIG / "bridge_precision_recall.png"); plt.close(fig)

# 5. Scenario stress outcomes using declared teaching assumptions.
exposure = 100_000_000
scenario_rows = []
for scenario in SCENARIOS:
    r = run_stress_test(exposure, scenario, n_paths=10000, seed=42)
    scenario_rows.append({k:r[k] for k in ["scenario","assumed_mean_impact_pct","expected_loss_usd","var_95_usd","var_99_usd","tvar_99_5_usd","simulated_gain_probability_pct"]})
    fig, ax = plt.subplots()
    vals = r["samples"] / 1_000_000
    ax.hist(vals, bins=60)
    ax.axvline(np.quantile(vals,.95), linestyle="--", label="VaR 95%")
    ax.set_xlabel("Simulated loss ($M; negative = gain)")
    ax.set_ylabel("Paths")
    ax.set_title(scenario)
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / ("stress_" + scenario.lower().replace(" ", "_").replace("-", "") + ".png")); plt.close(fig)
pd.DataFrame(scenario_rows).to_csv(ROOT / "outputs" / "stress_scenarios_100m.csv", index=False)

# 6. ESG example for steel HRC at 10,000 kg.
pair = carbon_pair_for_commodity(DATA["cross_domain"], "Steel_HRC")
esg = substitution_impact(10000, pair["carbon_virgin"], pair["carbon_recycled"], pair["green_premium"])
fig, ax = plt.subplots()
ax.bar(["Virgin", "Recycled"], [esg["baseline_emissions_kgco2e"]/1000, esg["substitute_emissions_kgco2e"]/1000])
ax.set_ylabel("Emissions (tonnes CO₂e)"); ax.set_title("Steel HRC sample intensity example (10,000 kg)")
fig.tight_layout(); fig.savefig(FIG / "esg_steel_substitution.png"); plt.close(fig)

summary = {
    "material_holdout": metrics["material_model"],
    "commodity_ablation": metrics["commodity_model"]["variants"],
    "bridge_holdout": {k:v for k,v in metrics["bridge_model"].items() if k != "classification_report_at_0_5"},
    "stress_scenarios_100m": scenario_rows,
    "esg_steel_10000kg_example": {k:v for k,v in esg.items() if k != "note"},
}
(ROOT / "outputs" / "result_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
print("Generated result plots and outputs/result_summary.json")
