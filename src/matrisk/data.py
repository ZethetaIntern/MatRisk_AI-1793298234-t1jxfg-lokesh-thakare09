"""Loading and validating the sample datasets supplied with the MatRisk AI brief."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw"

FILES = {
    "materials": "DS1_material_properties_5500.csv",
    "prices": "DS2_commodity_prices_10yr.csv",
    "bridges": "DS3_infrastructure_bridges_5000.csv",
    "cross_domain": "DS4_crossdomain_features_daily.csv",
    "element_prices": "DS5_element_prices_monthly.csv",
    "failures": "DS6_historical_failures_2000.csv",
}


def load_dataset(name: str, data_dir: Path | str | None = None) -> pd.DataFrame:
    """Load one named CSV, raising a helpful error when the file is missing."""
    if name not in FILES:
        raise KeyError(f"Unknown dataset '{name}'. Expected one of: {', '.join(FILES)}")
    base = Path(data_dir) if data_dir is not None else DATA_DIR
    path = base / FILES[name]
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Extract MatRisk-Data-Set-Sample.zip into data/raw/ first."
        )
    return pd.read_csv(path)


def load_all(data_dir: Path | str | None = None) -> dict[str, pd.DataFrame]:
    """Load all six provided datasets into a dictionary keyed by domain."""
    return {name: load_dataset(name, data_dir) for name in FILES}


def dataset_summary(data_dir: Path | str | None = None) -> pd.DataFrame:
    """Return row/column counts and missing-value percentages for each dataset."""
    rows = []
    for name, filename in FILES.items():
        frame = load_dataset(name, data_dir)
        rows.append({
            "dataset": name,
            "file": filename,
            "rows": int(frame.shape[0]),
            "columns": int(frame.shape[1]),
            "cells_missing_pct": round(100 * frame.isna().mean().mean(), 3),
        })
    return pd.DataFrame(rows)
