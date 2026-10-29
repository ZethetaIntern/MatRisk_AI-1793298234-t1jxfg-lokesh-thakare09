"""Dataset-level checks that make data limitations visible before modelling."""
from __future__ import annotations
from typing import Any
import pandas as pd


def quality_report(datasets: dict[str, pd.DataFrame]) -> dict[str, Any]:
    report: dict[str, Any] = {"datasets": {}, "checks": {}}
    for name, df in datasets.items():
        report["datasets"][name] = {
            "rows": int(df.shape[0]), "columns": int(df.shape[1]),
            "missing_cell_pct": round(100 * float(df.isna().mean().mean()), 3),
            "duplicate_rows": int(df.duplicated().sum()),
        }
    materials = datasets["materials"]
    bridges = datasets["bridges"]
    prices = datasets["prices"]
    cross = datasets["cross_domain"]
    report["checks"] = {
        "material_positive_K_pct": round(100 * float((materials["bulk_modulus_GPa"] > 0).mean()), 2),
        "material_positive_G_pct": round(100 * float((materials["shear_modulus_GPa"] > 0).mean()), 2),
        "material_density_positive_pct": round(100 * float((materials["density_g_cm3"] > 0).mean()), 2),
        "poisson_bounds_pct": round(100 * float(((materials["poisson_ratio"] > -1) & (materials["poisson_ratio"] < 0.5)).mean()), 2),
        "commodity_count": int(prices["commodity"].nunique()),
        "commodity_date_min": str(pd.to_datetime(prices["date"]).min().date()),
        "commodity_date_max": str(pd.to_datetime(prices["date"]).max().date()),
        "bridge_target_prevalence_pct": round(100 * float(bridges["structurally_deficient"].mean()), 2),
        "cross_domain_key_duplicates": int(cross.duplicated(["date", "commodity"]).sum()),
        "interpretation": "These CSVs were supplied as project sample datasets; they are treated as sample data, not independently verified live production data.",
    }
    return report
