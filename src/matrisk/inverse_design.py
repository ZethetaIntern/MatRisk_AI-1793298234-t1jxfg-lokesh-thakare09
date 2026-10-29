"""Property-and-cost screening against known materials (not a generative GAN)."""
from __future__ import annotations
import re
from typing import Any
import pandas as pd

ATOMIC_MASS = {
    "H":1.008,"Li":6.94,"B":10.81,"C":12.011,"N":14.007,"O":15.999,"F":18.998,
    "Na":22.990,"Mg":24.305,"Al":26.982,"Si":28.085,"P":30.974,"S":32.06,"Cl":35.45,
    "K":39.098,"Ca":40.078,"Sc":44.956,"Ti":47.867,"V":50.942,"Cr":51.996,"Mn":54.938,
    "Fe":55.845,"Co":58.933,"Ni":58.693,"Cu":63.546,"Zn":65.38,"Ga":69.723,"Ge":72.630,
    "Se":78.971,"Br":79.904,"Zr":91.224,"Nb":92.906,"Mo":95.95,"Ag":107.868,"Cd":112.414,
    "Sn":118.710,"Sb":121.760,"Te":127.60,"I":126.904,"Ce":140.116,"La":138.905,"Nd":144.242,
    "Ta":180.948,"W":183.84,"Re":186.207,"Pt":195.084,"Au":196.967,"Pb":207.2,"Bi":208.980,
    "Dy":162.500,"Hf":178.49,"In":114.818,"Li":6.94,"Nb":92.906,"Re":186.207,"Ru":101.07,
    "Rh":102.906,"Pd":106.42,"Y":88.906,"Yb":173.045,"Er":167.259,"Gd":157.25,"Pr":140.908,
}


def parse_formula(formula: str) -> dict[str, float]:
    """Parse a simple, unparenthesized chemical formula into elemental atom counts."""
    parts = re.findall(r"([A-Z][a-z]?)(\d*(?:\.\d+)?)", str(formula))
    parsed = {el: float(count) if count else 1.0 for el, count in parts}
    rebuilt = "".join(el + (str(int(n)) if n.is_integer() and n != 1 else (str(n) if n != 1 else "")) for el, n in parsed.items())
    # Reject formulas the small parser did not fully consume (e.g., nested parentheses).
    if not parts or re.sub(r"[A-Z][a-z]?\d*(?:\.\d+)?", "", str(formula)):
        return {}
    return parsed


def elemental_cost_proxy(formula: str, prices: pd.DataFrame) -> float | None:
    """Approximate weighted elemental spot-price proxy in USD/kg; excludes processing cost."""
    comp = parse_formula(formula)
    if not comp or any(el not in ATOMIC_MASS for el in comp):
        return None
    px = prices.copy()
    px["date"] = pd.to_datetime(px["date"])
    latest = px.sort_values("date").groupby("element").tail(1).set_index("element")["price_usd_per_kg"].to_dict()
    if any(el not in latest for el in comp):
        return None
    masses = {el: atoms * ATOMIC_MASS[el] for el, atoms in comp.items()}
    total_mass = sum(masses.values())
    if total_mass <= 0:
        return None
    return float(sum((mass / total_mass) * float(latest[el]) for el, mass in masses.items()))


def screen_existing_materials(materials: pd.DataFrame, prices: pd.DataFrame, target_bulk_GPa: float, max_density_g_cm3: float, max_cost_proxy_usd_kg: float, top_n: int = 5) -> pd.DataFrame:
    """Rank existing records; this is a screening heuristic rather than generated-alloy design."""
    df = materials.copy()
    df["element_cost_proxy_usd_kg"] = df["formula"].map(lambda f: elemental_cost_proxy(f, prices))
    df = df.loc[(df["bulk_modulus_GPa"] > 0) & (df["density_g_cm3"] > 0)].copy()
    df["density_ok"] = df["density_g_cm3"] <= max_density_g_cm3
    df["cost_ok"] = df["element_cost_proxy_usd_kg"].notna() & (df["element_cost_proxy_usd_kg"] <= max_cost_proxy_usd_kg)
    df["bulk_match_score"] = 1.0 / (1.0 + (df["bulk_modulus_GPa"] - target_bulk_GPa).abs() / max(abs(target_bulk_GPa), 1.0))
    df["density_score"] = (1.0 - (df["density_g_cm3"] / max(max_density_g_cm3, 0.1)).clip(upper=2)).clip(lower=0)
    df["cost_score"] = (1.0 - (df["element_cost_proxy_usd_kg"] / max(max_cost_proxy_usd_kg, 0.1)).clip(upper=2)).clip(lower=0).fillna(0)
    df["screen_score"] = 0.55 * df["bulk_match_score"] + 0.25 * df["density_score"] + 0.20 * df["cost_score"]
    df["passes_user_constraints"] = df["density_ok"] & df["cost_ok"]
    return df.sort_values(["passes_user_constraints", "screen_score"], ascending=[False, False]).head(top_n)[[
        "material_id", "formula", "crystal_system", "bulk_modulus_GPa", "shear_modulus_GPa", "density_g_cm3",
        "element_cost_proxy_usd_kg", "density_ok", "cost_ok", "passes_user_constraints", "screen_score"
    ]]
