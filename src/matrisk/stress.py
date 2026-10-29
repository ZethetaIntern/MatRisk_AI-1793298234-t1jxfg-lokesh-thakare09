"""Monte Carlo stress tests and a simple learning-game scoring function.

Scenario transmission rates below are explicit teaching assumptions. They are not
estimated causal effects and should be recalibrated with validated exposure data.
"""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd

SCENARIOS: dict[str, dict[str, Any]] = {
    "Rare earth supply crisis": {
        "shock": "40% export reduction", "duration": "3-4 quarters",
        "transmission_loss_pct": 2.5, "uncertainty_pct": 1.7,
        "explanation": "Supply-constrained inputs can raise costs for dependent industrial equipment and energy assets.",
    },
    "Steel quality shift": {
        "shock": "15% tensile-strength drop", "duration": "2-3 quarters",
        "transmission_loss_pct": 6.0, "uncertainty_pct": 3.0,
        "explanation": "Lower strength can increase inspection, repair or replacement pressure for steel-heavy structures.",
    },
    "Climate corrosion": {
        "shock": "30% corrosion-rate increase", "duration": "Persistent",
        "transmission_loss_pct": 4.0, "uncertainty_pct": 2.5,
        "explanation": "Faster corrosion raises maintenance needs and can shorten useful life in exposed assets.",
    },
    "Recycled-material mandate": {
        "shock": "50% recycled content requirement", "duration": "4-6 quarters",
        "transmission_loss_pct": 1.5, "uncertainty_pct": 1.0,
        "explanation": "Compliance can create transition costs while reducing lifecycle emissions when substitutes are suitable.",
    },
    "Novel alloy discovery": {
        "shock": "20% stronger/cheaper incumbent substitute", "duration": "6-8 quarters",
        "transmission_loss_pct": -2.5, "uncertainty_pct": 2.0,
        "explanation": "A successful substitute can lower future repair costs but may reduce the value of incumbent commodity positions.",
    },
}


def run_stress_test(exposure_usd: float, scenario: str, severity_multiplier: float = 1.0, n_paths: int = 10_000, seed: int = 42) -> dict[str, Any]:
    """Simulate a transparent normal approximation around a scenario's assumed loss rate."""
    if exposure_usd < 0:
        raise ValueError("exposure_usd must be non-negative")
    if scenario not in SCENARIOS:
        raise KeyError(f"Unknown scenario: {scenario}")
    if not 0.5 <= severity_multiplier <= 2.0:
        raise ValueError("severity_multiplier must be between 0.5 and 2.0")
    if n_paths < 100:
        raise ValueError("Use at least 100 Monte Carlo paths")
    params = SCENARIOS[scenario]
    rng = np.random.default_rng(seed)
    mean_fraction = params["transmission_loss_pct"] / 100 * severity_multiplier
    sd_fraction = params["uncertainty_pct"] / 100 * severity_multiplier
    loss = rng.normal(mean_fraction, sd_fraction, n_paths) * exposure_usd
    q95 = float(np.quantile(loss, 0.95))
    q99 = float(np.quantile(loss, 0.99))
    q995 = float(np.quantile(loss, 0.995))
    return {
        "scenario": scenario,
        "exposure_usd": float(exposure_usd),
        "severity_multiplier": float(severity_multiplier),
        "n_paths": int(n_paths),
        "assumed_mean_impact_pct": round(mean_fraction * 100, 3),
        "expected_loss_usd": float(loss.mean()),
        "var_95_usd": q95,
        "var_99_usd": q99,
        "tvar_99_5_usd": float(loss[loss >= q995].mean()),
        "max_simulated_loss_usd": float(loss.max()),
        "simulated_gain_probability_pct": float(100 * (loss < 0).mean()),
        "assumption_note": "Monte Carlo outputs depend on illustrative mean/uncertainty rates defined in stress.py; not calibrated forecasts.",
        "samples": loss,
    }


def game_outcome(scenario: str, action: str, exposure_usd: float = 10_000_000.0, severity_multiplier: float = 1.0, seed: int = 42) -> dict[str, Any]:
    """Score one user decision in MatRisk Lab campaign levels 1-2."""
    base = run_stress_test(exposure_usd, scenario, severity_multiplier, n_paths=5000, seed=seed)
    mitigation = {
        "Increase maintenance / inspection": 0.55 if scenario in {"Climate corrosion", "Steel quality shift"} else 0.12,
        "Reduce exposed commodity position": 0.55 if scenario in {"Rare earth supply crisis", "Steel quality shift", "Novel alloy discovery"} else 0.25,
        "Diversify materials and suppliers": 0.35,
        "Do nothing": 0.0,
    }
    rate = mitigation.get(action)
    if rate is None:
        raise KeyError(f"Unknown action: {action}")
    # Immediate intervention costs are assumptions and included so that mitigation is not free.
    cost_fraction = {
        "Increase maintenance / inspection": 0.010,
        "Reduce exposed commodity position": 0.006,
        "Diversify materials and suppliers": 0.008,
        "Do nothing": 0.0,
    }[action]
    gross_expected_loss = base["expected_loss_usd"]
    avoided_loss = max(gross_expected_loss, 0.0) * rate
    action_cost = exposure_usd * cost_fraction
    net_loss = gross_expected_loss - avoided_loss + action_cost
    score = int(np.clip(1000 - max(net_loss, 0) / max(exposure_usd, 1) * 6000, 0, 1000))
    score += 120 if rate >= 0.5 else (70 if rate >= 0.3 else 20)
    score = int(min(1000, score))
    return {
        "scenario": scenario, "action": action,
        "baseline_expected_loss_usd": gross_expected_loss,
        "avoided_loss_usd": float(avoided_loss), "action_cost_usd": float(action_cost),
        "net_expected_loss_usd": float(net_loss), "mitigation_fraction": rate,
        "score_out_of_1000": score,
        "feedback": _feedback(scenario, action, rate),
        "assumption_note": base["assumption_note"],
    }


def _feedback(scenario: str, action: str, mitigation: float) -> str:
    if mitigation >= 0.5:
        return f"Strong response to {scenario}: your choice directly addresses the main exposure in this simplified game."
    if mitigation >= 0.3:
        return f"Reasonable portfolio-level response. Consider whether targeted maintenance or a hedge is more specific to {scenario}."
    if action == "Do nothing":
        return "No mitigation was applied. Compare the saved intervention cost with the risk left in the portfolio."
    return "This action gives only limited protection in the current scenario. Explain the trade-off before choosing it in a real decision." 
