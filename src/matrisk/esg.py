"""Simple carbon and cost impact calculations for material substitutions."""
from __future__ import annotations
from typing import Any
import pandas as pd


def substitution_impact(quantity_kg: float, carbon_virgin_kgco2e_per_kg: float, carbon_recycled_kgco2e_per_kg: float, green_premium_usd_per_kg: float = 0.0) -> dict[str, Any]:
    """Calculate avoided emissions and incremental cost for a full virgin-to-recycled switch."""
    if quantity_kg < 0:
        raise ValueError("quantity_kg must be non-negative")
    if carbon_virgin_kgco2e_per_kg < 0 or carbon_recycled_kgco2e_per_kg < 0:
        raise ValueError("carbon intensity inputs must be non-negative")
    avoided = (carbon_virgin_kgco2e_per_kg - carbon_recycled_kgco2e_per_kg) * quantity_kg
    return {
        "quantity_kg": float(quantity_kg),
        "baseline_emissions_kgco2e": float(quantity_kg * carbon_virgin_kgco2e_per_kg),
        "substitute_emissions_kgco2e": float(quantity_kg * carbon_recycled_kgco2e_per_kg),
        "avoided_emissions_kgco2e": float(avoided),
        "avoided_emissions_tco2e": float(avoided / 1000),
        "incremental_cost_usd": float(quantity_kg * green_premium_usd_per_kg),
        "cost_per_tco2e_avoided_usd": float((quantity_kg * green_premium_usd_per_kg) / (avoided / 1000.0)) if avoided > 0 else None,
        "note": "This is a scenario calculation. Actual lifecycle values depend on supplier, electricity mix, transport and system boundaries.",
    }


def carbon_pair_for_commodity(cross_domain: pd.DataFrame, commodity: str) -> dict[str, float]:
    """Use latest available cross-domain carbon and green-premium values for one commodity."""
    df = cross_domain.loc[cross_domain["commodity"] == commodity].copy()
    if df.empty:
        raise KeyError(f"Commodity not found: {commodity}")
    df["date"] = pd.to_datetime(df["date"])
    row = df.sort_values("date").iloc[-1]
    return {
        "carbon_virgin": float(row["carbon_intensity_virgin"]),
        "carbon_recycled": float(row["carbon_intensity_recycled"]),
        "green_premium": float(row["green_premium_per_kg"]),
    }
