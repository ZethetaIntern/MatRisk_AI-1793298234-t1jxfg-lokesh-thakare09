"""Streamlit interface for the MatRisk AI student prototype.

Run from the project root with: streamlit run app.py
"""
from __future__ import annotations
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from matrisk.data import load_all, DATA_DIR  # noqa: E402
from matrisk.material_model import predict_material, TARGETS, TARGET_LABELS  # noqa: E402
from matrisk.commodity_model import predict_latest_commodity_signal, TRADITIONAL_FEATURES, CROSS_DOMAIN_FEATURES  # noqa: E402
from matrisk.bridge_risk import financial_risk_metrics  # noqa: E402
from matrisk.stress import SCENARIOS, run_stress_test, game_outcome  # noqa: E402
from matrisk.esg import carbon_pair_for_commodity, substitution_impact  # noqa: E402
from matrisk.inverse_design import screen_existing_materials  # noqa: E402

st.set_page_config(page_title="MatRisk AI | Material-Financial Intelligence", page_icon="🧪", layout="wide")

@st.cache_data(show_spinner=False)
def get_data():
    return load_all()

@st.cache_resource(show_spinner=False)
def get_model(name: str):
    path = ROOT / "models" / name
    return joblib.load(path) if path.exists() else None

@st.cache_data(show_spinner=False)
def get_metrics():
    path = ROOT / "outputs" / "metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

try:
    data = get_data()
except Exception as exc:
    st.error(f"Unable to load supplied sample data: {exc}")
    st.stop()
metrics = get_metrics()
page = st.sidebar.radio("Navigate", [
    "Overview", "Material Property Explorer", "Commodity Signals", "Infrastructure Risk",
    "Inverse Material Screening", "ESG Impact", "MatRisk Lab Game"
])
st.sidebar.caption("Student prototype • sample datasets • not a production financial or engineering tool")

if not metrics:
    st.warning("The model files are not trained yet. From this folder run: `python -m matrisk.train_all` (after installing requirements).")

if page == "Overview":
    st.title("MatRisk AI")
    st.subheader("Material intelligence connected to financial decisions")
    st.write("This prototype joins material-property data, commodity-price histories, bridge conditions and cross-domain indicators. It demonstrates baseline modelling and scenario analysis; it is not a validated production risk engine.")
    cols = st.columns(4)
    kpis = [("Material records", len(data["materials"])), ("Commodity observations", len(data["prices"])), ("Bridge records", len(data["bridges"])), ("Failure events", len(data["failures"]))]
    for c, (label, value) in zip(cols, kpis):
        c.metric(label, f"{value:,}")
    st.markdown("### What runs in this prototype")
    st.markdown("""
    - **Material model:** ExtraTrees predicts four material properties from formula strings and tabular descriptors.
    - **Commodity model:** a chronological return baseline compares technical features with technical plus material-financial features.
    - **Infrastructure model:** a classifier estimates structural-deficiency risk, then illustrative PD-proxy × LGD-proxy × EAD calculations are shown.
    - **Stress and ESG:** assumptions are exposed so the user can change scenario severity and substitution quantity.
    - **MatRisk Lab:** two playable learning levels with a decision score.
    """)
    if metrics:
        st.markdown("### Observed test metrics")
        mat = metrics.get("material_model", {}).get("per_target", {})
        st.dataframe(pd.DataFrame([{"Target": v.get("label", k), "MAE": v.get("mae"), "RMSE": v.get("rmse"), "R²": v.get("r2")} for k,v in mat.items()]), use_container_width=True, hide_index=True)
        com = metrics.get("commodity_model", {}).get("variants", {})
        if com:
            st.dataframe(pd.DataFrame([{"Model": k.replace("_", " "), "MAE": v.get("mae_next_day_log_return"), "Direction accuracy (%)": v.get("directional_accuracy_pct"), "Toy strategy Sharpe": v.get("strategy_sharpe_annualised")} for k,v in com.items()]), use_container_width=True, hide_index=True)
    st.info("Use the results as a prototype benchmark. The input CSVs are provided sample data and were not independently verified as live production records.")

elif page == "Material Property Explorer":
    st.title("Material Property Explorer")
    model = get_model("material_extratrees.joblib")
    if model is None:
        st.error("Train models first with `python -m matrisk.train_all`.")
        st.stop()
    materials = data["materials"]
    options = materials["formula"].astype(str) + " | " + materials["material_id"].astype(str)
    idx = st.selectbox("Select a sample material", options.index, format_func=lambda i: options.loc[i])
    record = materials.loc[idx].copy()
    left, right = st.columns([1, 1])
    with left:
        st.write("**Material metadata**")
        st.write({"Material ID": record["material_id"], "Formula": record["formula"], "Crystal system": record["crystal_system"], "Category": record["category"], "Space group": int(record["spacegroup_number"])})
        st.caption("The current dataset has descriptors, not full atomic coordinates. A real crystal graph cannot be reconstructed from this CSV alone.")
    pred = predict_material(model, record)
    with right:
        st.write("**Model prediction vs. supplied value**")
        display_rows = []
        for target in TARGETS:
            display_rows.append({"Property": TARGET_LABELS[target], "Actual sample value": record[target], "Model prediction": pred[target], "Absolute error": abs(float(record[target]) - pred[target])})
        st.dataframe(pd.DataFrame(display_rows).round(4), use_container_width=True, hide_index=True)
    fig = go.Figure()
    for target in TARGETS:
        fig.add_trace(go.Bar(name=TARGET_LABELS[target], x=[TARGET_LABELS[target]], y=[pred[target]]))
    fig.update_layout(title="Predicted properties (units shown in labels)", showlegend=False, xaxis_title="Property", yaxis_title="Predicted value")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("### Physics check")
    pcols = st.columns(4)
    pcols[0].metric("Derived Poisson ratio", f"{pred['poisson_ratio_derived']:.3f}")
    pcols[1].metric("Young's modulus from G", f"{pred['young_modulus_from_shear_GPa']:.2f} GPa")
    pcols[2].metric("Young's modulus from K", f"{pred['young_modulus_from_bulk_GPa']:.2f} GPa")
    pcols[3].metric("Elastic consistency gap", f"{pred['elastic_relative_gap_pct']:.5f}%")
    st.caption("Poisson's ratio is derived algebraically from predicted K and G so the displayed isotropic elastic relationship is consistent. This is a post-processing constraint, not a trained PINN.")
    if metrics:
        st.markdown("### Holdout evaluation by chemical system")
        pm = metrics["material_model"]["per_target"]
        st.dataframe(pd.DataFrame([{"Target": v["label"], "Holdout MAE": v["mae"], "Holdout RMSE": v["rmse"], "R²": v["r2"]} for v in pm.values()]), hide_index=True, use_container_width=True)

elif page == "Commodity Signals":
    st.title("Commodity Signal Dashboard")
    full_model = get_model("commodity_full_data_for_demo.joblib")
    if full_model is None:
        st.error("Train models first with `python -m matrisk.train_all`.")
        st.stop()
    feature_columns = metrics.get("commodity_model", {}).get("feature_columns", {}).get("full_data_for_demo", [])
    signals = predict_latest_commodity_signal(full_model, data["prices"], data["cross_domain"], feature_columns)
    chosen = st.selectbox("Commodity", signals["commodity"].sort_values().tolist())
    row = signals.loc[signals["commodity"] == chosen].iloc[0]
    cols = st.columns(4)
    cols[0].metric("Latest sample close", f"${row['close']:,.2f}")
    cols[1].metric("Predicted next-day log return", f"{row['predicted_next_day_log_return']:.4%}")
    cols[2].metric("MQI", f"{row['mqi']:.1f}/100")
    cols[3].metric("Supply disruption probability", f"{row['supply_disruption_prob']:.1%}")
    price = data["prices"].loc[data["prices"]["commodity"] == chosen].copy()
    price["date"] = pd.to_datetime(price["date"])
    price = price.sort_values("date").tail(500)
    fig = go.Figure(data=[go.Candlestick(x=price["date"], open=price["open"], high=price["high"], low=price["low"], close=price["close"], name=chosen)])
    fig.update_layout(title=f"{chosen}: recent sample price history", xaxis_title="Date", yaxis_title="Price (sample dataset units)", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
    cross = data["cross_domain"].loc[data["cross_domain"]["commodity"] == chosen].copy()
    cross["date"] = pd.to_datetime(cross["date"])
    fig2 = px.line(cross.tail(250), x="date", y="mqi", title="Material Quality Index (MQI) — sample series")
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("### Out-of-time ablation")
    if metrics:
        variants = metrics.get("commodity_model", {}).get("variants", {})
        st.dataframe(pd.DataFrame([{"Model variant": k.replace("_", " "), **{x: v.get(x) for x in ["n_test", "mae_next_day_log_return", "directional_accuracy_pct", "strategy_sharpe_annualised", "strategy_total_log_return_sum", "max_drawdown_proxy_pct"]}} for k,v in variants.items()]), use_container_width=True, hide_index=True)
    st.caption("The toy strategy is only an evaluation example. A positive Sharpe in this synthetic/sample environment is not proof of investable alpha; transaction cost and portfolio construction assumptions matter.")

elif page == "Infrastructure Risk":
    st.title("Infrastructure Risk Scorer")
    model = get_model("bridge_deficiency_rf.joblib")
    if model is None:
        st.error("Train models first with `python -m matrisk.train_all`.")
        st.stop()
    risk = financial_risk_metrics(data["bridges"], model, data["failures"])
    risk = risk.sort_values("pd_proxy", ascending=False)
    chosen_id = st.selectbox("Bridge asset", risk["bridge_id"].astype(str).tolist(), index=0)
    row = risk.loc[risk["bridge_id"].astype(str) == chosen_id].iloc[0]
    c = st.columns(4)
    c[0].metric("Deficiency risk proxy", f"{row['pd_proxy']:.1%}")
    c[1].metric("LGD proxy", f"{row['lgd_proxy']:.1%}")
    c[2].metric("Exposure at default", f"${row['ead_M']:.2f}M")
    c[3].metric("Expected loss proxy", f"${row['expected_loss_proxy_M']:.3f}M")
    st.write({"Bridge type": row["bridge_type"], "Material": row["material"], "Age": int(row["age_years"]), "Condition rating": int(row["condition_rating"]), "Corrosion rate (mm/year)": row["corrosion_rate_mm_yr"], "Risk band": str(row["risk_band"])})
    fig = px.scatter(risk.sample(min(1200, len(risk)), random_state=42), x="age_years", y="pd_proxy", color="risk_band", size="loan_outstanding_M", hover_data=["bridge_id", "material", "condition_rating"], title="Deficiency-risk proxy by asset age")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("### Highest-risk assets in the supplied sample")
    st.dataframe(risk[["bridge_id", "bridge_type", "material", "age_years", "condition_rating", "pd_proxy", "lgd_proxy", "ead_M", "expected_loss_proxy_M", "risk_band"]].head(20).round(4), use_container_width=True, hide_index=True)
    if metrics:
        bm = metrics.get("bridge_model", {})
        st.markdown(f"**Holdout ROC-AUC:** {bm.get('roc_auc')} · **Average precision:** {bm.get('average_precision')} · **Test prevalence:** {bm.get('positive_prevalence_pct_test')}%")
    st.warning("The classifier estimates structural-deficiency status, not a calibrated annual probability of loan default. PD/LGD/EL values are illustrative proxies.")

elif page == "Inverse Material Screening":
    st.title("Inverse Material Designer — Screening Prototype")
    st.write("The brief proposes a WGAN-GP to generate new alloys. The supplied sample lacks the full atomic structures and trained property predictor required to validate generated alloys, so this page ranks existing records against a target profile instead.")
    c1,c2,c3 = st.columns(3)
    target_bulk = c1.slider("Target bulk modulus (GPa)", min_value=10, max_value=400, value=150, step=5)
    max_density = c2.slider("Maximum density (g/cm³)", min_value=1.0, max_value=20.0, value=8.0, step=0.5)
    max_cost = c3.slider("Maximum elemental cost proxy (USD/kg)", min_value=1.0, max_value=500.0, value=80.0, step=5.0)
    if st.button("Screen known materials", type="primary"):
        candidates = screen_existing_materials(data["materials"], data["element_prices"], target_bulk, max_density, max_cost, top_n=10)
        st.dataframe(candidates.round(4), use_container_width=True, hide_index=True)
        st.caption("Cost proxy is a mass-weighted blend of latest sample elemental prices; it excludes processing, manufacturing, certification and supply-chain cost.")
        fig = px.scatter(candidates, x="element_cost_proxy_usd_kg", y="bulk_modulus_GPa", color="passes_user_constraints", hover_name="formula", size="density_g_cm3", title="Candidate screening: cost proxy vs. bulk modulus")
        st.plotly_chart(fig, use_container_width=True)

elif page == "ESG Impact":
    st.title("ESG Impact Analyser")
    commodity = st.selectbox("Material/commodity reference", sorted(data["cross_domain"]["commodity"].unique()))
    pair = carbon_pair_for_commodity(data["cross_domain"], commodity)
    quantity = st.number_input("Quantity to substitute (kg)", min_value=0.0, value=10000.0, step=1000.0)
    c1,c2,c3 = st.columns(3)
    c1.metric("Virgin intensity", f"{pair['carbon_virgin']:.3f} kg CO₂e/kg")
    c2.metric("Recycled intensity", f"{pair['carbon_recycled']:.3f} kg CO₂e/kg")
    c3.metric("Green premium", f"${pair['green_premium']:.3f}/kg")
    result = substitution_impact(quantity, pair["carbon_virgin"], pair["carbon_recycled"], pair["green_premium"])
    cols = st.columns(4)
    cols[0].metric("Baseline emissions", f"{result['baseline_emissions_kgco2e']:,.0f} kg")
    cols[1].metric("Substitute emissions", f"{result['substitute_emissions_kgco2e']:,.0f} kg")
    cols[2].metric("Avoided emissions", f"{result['avoided_emissions_tco2e']:,.2f} tCO₂e")
    cols[3].metric("Incremental cost", f"${result['incremental_cost_usd']:,.2f}")
    st.write({"Cost per tonne CO₂e avoided (USD)": result["cost_per_tco2e_avoided_usd"], "Note": result["note"]})
    st.warning("Carbon values and green premiums come from the supplied sample dataset. Verify units and lifecycle boundaries before real ESG reporting.")

elif page == "MatRisk Lab Game":
    st.title("MatRisk Lab — Campaign Mode")
    st.write("You are managing an infrastructure and commodity portfolio. Choose a response to a material-related shock, then compare the reduction in simulated expected loss with the intervention cost.")
    level = st.radio("Campaign level", ["Level 1 — Monitor a steel bridge and steel position", "Level 2 — Respond to a material shock"], horizontal=False)
    if level.startswith("Level 1"):
        scenario = "Steel quality shift"
        st.info("Level 1 introduces portfolio monitoring. A steel quality signal appears, and you can choose whether to reduce exposure, inspect/maintain or hold your position.")
    else:
        scenario = st.selectbox("Material-financial shock", list(SCENARIOS.keys()))
    action = st.selectbox("Your response", ["Increase maintenance / inspection", "Reduce exposed commodity position", "Diversify materials and suppliers", "Do nothing"])
    exposure = st.number_input("Portfolio exposure (USD)", min_value=100000.0, max_value=500000000.0, value=10000000.0, step=500000.0)
    severity = st.slider("Shock severity multiplier", 0.5, 2.0, 1.0, 0.1)
    if st.button("Resolve scenario", type="primary"):
        result = game_outcome(scenario, action, exposure, severity)
        c = st.columns(4)
        c[0].metric("Score", f"{result['score_out_of_1000']}/1000")
        c[1].metric("Baseline expected loss", f"${result['baseline_expected_loss_usd']:,.0f}")
        c[2].metric("Avoided loss", f"${result['avoided_loss_usd']:,.0f}")
        c[3].metric("Net expected loss", f"${result['net_expected_loss_usd']:,.0f}")
        st.success(result["feedback"])
        st.write({"Mitigation fraction": result["mitigation_fraction"], "Action cost": result["action_cost_usd"], "Assumption": result["assumption_note"]})
        st.caption("Game scores and impact multipliers are teaching assumptions; they do not represent investment advice or calibrated engineering risk.")

st.sidebar.divider()
st.sidebar.caption("Built using the six supplied DS-1 to DS-6 sample CSV files. Review README and viva notes before presenting.")
