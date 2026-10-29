"""Train all prototype baselines and write a reproducible metrics bundle."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import pandas as pd
from .data import DATA_DIR, load_all
from .material_model import train_material_model
from .commodity_model import train_commodity_models
from .bridge_risk import train_bridge_model
from .quality import quality_report

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"


def main() -> dict:
    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    data = load_all()
    metrics = {"project": "MatRisk AI prototype", "data_quality": quality_report(data)}
    _, metrics["material_model"] = train_material_model(data["materials"], MODEL_DIR / "material_extratrees.joblib")
    _, metrics["commodity_model"] = train_commodity_models(data["prices"], data["cross_domain"], MODEL_DIR)
    _, metrics["bridge_model"] = train_bridge_model(data["bridges"], MODEL_DIR / "bridge_deficiency_rf.joblib")
    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    # Small data dictionary makes the bundled project self-describing.
    dataset_rows = []
    for name, df in data.items():
        dataset_rows.append({"dataset": name, "rows": len(df), "columns": len(df.columns), "missing_cell_pct": round(100*df.isna().mean().mean(), 3)})
    pd.DataFrame(dataset_rows).to_csv(OUTPUT_DIR / "dataset_summary.csv", index=False)
    print(json.dumps({
        "data_quality": metrics["data_quality"],
        "material_metrics": metrics["material_model"]["per_target"],
        "commodity_metrics": metrics["commodity_model"]["variants"],
        "bridge_metrics": {k:v for k,v in metrics["bridge_model"].items() if k in ["roc_auc", "average_precision", "brier_score", "positive_prevalence_pct_test"]},
    }, indent=2))
    return metrics

if __name__ == "__main__":
    main()
