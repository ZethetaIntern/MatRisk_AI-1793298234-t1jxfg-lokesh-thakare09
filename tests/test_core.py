from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from matrisk.data import load_all
from matrisk.material_model import chemical_system, predict_material
from matrisk.commodity_model import prepare_commodity_frame
from matrisk.stress import SCENARIOS, run_stress_test, game_outcome
from matrisk.esg import substitution_impact
from matrisk.inverse_design import parse_formula


def test_sample_data_has_expected_domains():
    data = load_all(ROOT / "data" / "raw")
    assert len(data["materials"]) == 5500
    assert data["prices"]["commodity"].nunique() == 8
    assert len(data["bridges"]) == 5000
    assert len(data["failures"]) == 2000


def test_chemical_system_groups_formula_variants():
    assert chemical_system("Fe2O3") == chemical_system("FeO") == "Fe-O"
    assert chemical_system("NaCl") == "Cl-Na"


def test_commodity_target_is_next_day_return_not_current_day():
    prices = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "commodity": ["Steel_HRC"] * 3,
        "daily_return": [0.01, -0.02, 0.03],
    })
    cross = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "commodity": ["Steel_HRC"] * 3,
        "mqi": [80, 81, 82],
    })
    out = prepare_commodity_frame(prices, cross)
    assert out.loc[0, "target_next_return"] == -0.02
    assert out.loc[1, "target_next_return"] == 0.03
    assert np.isnan(out.loc[2, "target_next_return"])


def test_stress_is_reproducible_and_has_expected_metrics():
    a = run_stress_test(1_000_000, "Climate corrosion", n_paths=1000, seed=10)
    b = run_stress_test(1_000_000, "Climate corrosion", n_paths=1000, seed=10)
    assert a["expected_loss_usd"] == b["expected_loss_usd"]
    assert a["var_95_usd"] > a["expected_loss_usd"]
    assert a["tvar_99_5_usd"] >= a["var_99_usd"]


def test_stress_validation():
    with pytest.raises(KeyError):
        run_stress_test(1000, "unknown")
    with pytest.raises(ValueError):
        run_stress_test(-1, "Climate corrosion")
    with pytest.raises(ValueError):
        run_stress_test(1000, "Climate corrosion", severity_multiplier=3)


def test_game_returns_score_and_decision_feedback():
    outcome = game_outcome("Climate corrosion", "Increase maintenance / inspection", 2_000_000)
    assert 0 <= outcome["score_out_of_1000"] <= 1000
    assert outcome["avoided_loss_usd"] >= 0
    assert outcome["feedback"]


def test_esg_substitution_math():
    out = substitution_impact(1000, 10.0, 2.0, 0.5)
    assert out["baseline_emissions_kgco2e"] == 10000
    assert out["substitute_emissions_kgco2e"] == 2000
    assert out["avoided_emissions_kgco2e"] == 8000
    assert out["incremental_cost_usd"] == 500
    assert out["cost_per_tco2e_avoided_usd"] == 62.5  # USD per tonne, not per kg
    with pytest.raises(ValueError):
        substitution_impact(-1, 10, 2)


def test_simple_formula_parser():
    assert parse_formula("Fe2O3") == {"Fe": 2.0, "O": 3.0}
    assert parse_formula("NaCl") == {"Na": 1.0, "Cl": 1.0}
    assert parse_formula("(FeO)2") == {}


def test_physics_postprocessing_produces_consistent_young_modulus():
    class DummyModel:
        def predict(self, X):
            return np.array([[-1.1, 1.2, 150.0, 60.0]])
    record = {"formula":"FeO", "n_elements":2, "spacegroup_number":225, "nsites":4,
              "volume_A3":50, "crystal_system":"cubic", "category":"oxide"}
    p = predict_material(DummyModel(), record)
    assert p["physics_bounds_valid"]
    assert p["poisson_ratio_derived"] > -1 and p["poisson_ratio_derived"] < 0.5
    assert p["elastic_relative_gap_pct"] < 1e-6


def test_stress_scenario_set_matches_brief_concept():
    assert len(SCENARIOS) == 5
    assert any("corrosion" in x.lower() for x in SCENARIOS)
