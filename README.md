# MatRisk AI — Material-Financial Risk Prototype

**Project type:** Data Science internship prototype  
**Language:** Python 3.11+  
**Interface:** Streamlit dashboard + FastAPI endpoints  
**Data:** The six sample CSV files supplied with the assignment

## 1. Problem in one sentence

MatRisk AI connects **what is happening to a material** (strength, stiffness, corrosion and substitute options) with **the financial decisions that depend on it** (commodity exposure, infrastructure repair risk, expected loss and carbon cost).

A simple example is a steel bridge. If the steel quality declines or corrosion accelerates, the owner may need to inspect or repair the bridge earlier. Those repairs affect the asset's remaining useful life, insurance costs and the lender's expected loss. MatRisk AI is a prototype for organising and testing that chain of reasoning.

## 2. What is implemented here

| Module | Implementation in this repository | Important boundary |
|---|---|---|
| Material property prediction | ExtraTrees multi-output baseline for formation energy, band gap, bulk modulus and shear modulus | This is **not** the assignment's Crystal Graph Neural Network (CGNN). The sample CSV has no atomic coordinates or full crystal structures. |
| Physics check | Bounds checks plus an algebraic post-processing step that derives Poisson's ratio from predicted bulk and shear moduli | This is **not** a trained PINN. The raw-data audit checks whether supplied K, G and ν values satisfy the isotropic elastic identity within a 10% tolerance. |
| Commodity signals | Historical technical-feature baseline compared with a technical + material-financial feature model; next-day returns and a chronological holdout | Out-of-time results are reported as observed. The sample dataset and simple directional strategy are not evidence of investable alpha. |
| Infrastructure risk | Random-forest classifier for `structurally_deficient`; financial arithmetic for PD-proxy × LGD-proxy × EAD | The classifier estimates the supplied deficiency label, not a calibrated annual probability of loan default or a survival curve. |
| Stress engine | 10,000-path Monte Carlo loss distribution for five assignment scenarios | Transmission rates are transparent teaching assumptions, not empirically calibrated causal estimates. |
| Inverse material design | Rank existing sample materials against bulk-modulus, density and elemental-cost-proxy targets | This is **screening**, not a WGAN-GP generating novel alloys. |
| ESG | Carbon-intensity substitution and incremental-cost calculator | Verify lifecycle assessment boundaries and units before using externally. |
| MatRisk Lab | Two campaign levels with scenario selection, action choice, intervention cost and score | The game is educational and its multipliers/scoring are assumptions. |
| Interface/API | Streamlit pages and basic FastAPI material/stress endpoints | Run locally for a demonstration; production security and monitoring are not implemented. |

The assignment itself describes a much larger 27-deliverable programme. This repository is a runnable prototype foundation and explicitly does not claim to have completed all specialist model requirements or external data ingestion.

## 3. Data supplied with the project

All six source files are in `data/raw/`.

| File | Rows | Intended use |
|---|---:|---|
| `DS1_material_properties_5500.csv` | 5,500 | Material property prediction and physics checks |
| `DS2_commodity_prices_10yr.csv` | 22,952 | Commodity time-series features and holdout evaluation |
| `DS3_infrastructure_bridges_5000.csv` | 5,000 | Bridge deficiency risk and exposure calculations |
| `DS4_crossdomain_features_daily.csv` | 22,952 | MQI, supply disruption, substitution, green premium and carbon indicators |
| `DS5_element_prices_monthly.csv` | 5,280 | Element-price cost proxy for material screening |
| `DS6_historical_failures_2000.csv` | 2,000 | Empirical sample loss-ratio distribution for illustrative LGD proxy |

These are the sample datasets provided for this assignment. They are not independently verified live market/engineering data. All model results must be read in that context.

## 4. Quick start (Windows PowerShell)

1. Install Python 3.11 or newer.
2. Open a terminal in this folder.
3. Create and activate a virtual environment:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Train the local baseline models and write `outputs/metrics.json`:

   ```powershell
   $env:PYTHONPATH = "src"
   python -m matrisk.train_all
   ```

   Or run `make train` on a machine with GNU Make.

5. Start the dashboard in a second terminal (same virtual environment):

   ```powershell
   streamlit run app.py
   ```

6. Open `http://localhost:8501`.

7. Optional API:

   ```powershell
   $env:PYTHONPATH = "src"
   uvicorn matrisk.api:app --reload --port 8000
   ```

   Then open `http://localhost:8000/docs`. Example endpoint: `GET /material/mp-121959`; stress endpoint: `POST /stress`.

## 5. Training and evaluation design

### Materials

- A group holdout is formed using the set of elements in each formula (chemical system). The point is to test on chemical systems not seen during training, reducing the risk of optimistic results from near-related compounds appearing in both splits.
- The model uses formula character n-grams, crystal system, broad category, site count, volume and element-count descriptors.
- The holdout reports MAE, RMSE and R² for each property.
- The physics audit is separate from the predictive error metrics. Predicted Poisson's ratio is derived as

  `nu = (3K - 2G) / (2(3K + G))`

  for positive bulk modulus `K` and shear modulus `G`, which makes `E = 2G(1 + nu) = 3K(1 - 2nu)` algebraically consistent for the displayed isotropic approximation.

### Commodities

- Price features and material-financial indicators are joined on `(date, commodity)`.
- The target is the *next* trading day's return, created by shifting the observed return within each commodity.
- Training uses dates before 2023-01-01. Test observations are from 2023 onward.
- The ablation compares technical indicators alone with technical plus cross-domain indicators.
- Directional accuracy, MAE, a toy strategy Sharpe ratio, cumulative log-return sum and a drawdown proxy are generated. No target metric (e.g. Sharpe > 0.5) is claimed until the actual holdout output demonstrates it.

### Infrastructure risk

- The model predicts `structurally_deficient` using age, material, corrosion, traffic, strength and thickness-related features. `condition_rating` is not supplied as a predictor.
- The random holdout is stratified, but the positive class is rare. Read average precision and the test prevalence together with ROC-AUC.
- The shown `PD proxy` equals the predicted deficiency probability. It is **not** an actuarially validated default probability over a specified time horizon.
- `Expected loss proxy = PD proxy × LGD proxy × EAD`. LGD is a sample loss-ratio mean, adjusted with a small condition heuristic for the demo.

## 6. Repository guide

```text
app.py                      Streamlit dashboard
src/matrisk/data.py          Dataset loader and dictionary
src/matrisk/material_model.py Material baseline + physics audit
src/matrisk/commodity_model.py Time-split commodity ablation
src/matrisk/bridge_risk.py   Deficiency classifier + EL arithmetic
src/matrisk/stress.py        Scenario Monte Carlo + game scoring
src/matrisk/esg.py           Carbon substitution calculations
src/matrisk/inverse_design.py Existing-material screening
src/matrisk/api.py           FastAPI wrapper
src/matrisk/train_all.py     Reproducible baseline training
notebooks/                  EDA notebooks (15+ charts each)
tests/                       Unit checks
configs/                     Model assumptions
reports/                     Technical report and result summary
docs/                        Architecture notes
outputs/                     Generated metrics, tables and figures
models/                      Trained local model artifacts (created by training)
```

## 7. Quality checks

```powershell
$env:PYTHONPATH = "src"
pytest -q
python -m compileall -q src app.py tests
```

For Docker, build and run `docker compose up --build`. This starts Streamlit, the FastAPI service and an MLflow UI container. The current training script writes local metrics and models; adding experiment-level MLflow logging and configuring a secure DVC remote remain follow-up engineering tasks.

## 8. Data versioning and deployment caveats

A `dvc.yaml` stage is provided as a starting point. To meet the assignment's full DVC requirement, install DVC, track the CSVs (`dvc add data/raw/*.csv`), configure an authorised remote, and push the data. Do not claim DVC data is backed up until `dvc push` succeeds. Do not commit credentials; use `.env.example` only as a template.

This code does not fetch 100,000+ Materials Project/AFLOW records or the full National Bridge Inventory. Such downloads require API/data access, licensing checks, data-cleaning and time to validate. The submitted sample files are used here so the project runs without external credentials.

## 9. Before presenting or submitting

- Run the pipeline on your computer and inspect `outputs/metrics.json` yourself.
- Change one or two model choices, rerun, and write down why the metric changed.
- Read `reports/viva_notes.md` and be ready to explain the target, split strategy, limitations and each formula.
- Add your own observations, screenshots and commit history after you run it.
- Confirm the internship's rules for AI-assisted work and follow the disclosure requirement, if any. Do not describe unimplemented components as completed.
- Follow the actual repository transfer procedure only after the supervisor confirms the destination and the ownership terms.

## 10. Sources and scientific context

See `reports/technical_report.md` for the literature list, including crystal graph learning, physics-informed learning, survival analysis, explainability and official material/infrastructure data documentation.

### Test coverage snapshot

On the local project test run, `pytest -q --cov=matrisk --cov-report=term-missing` passed **19 tests** and reported **97% statement coverage across `src/matrisk`**. This does not mean every user-interface interaction or deployment path has been validated. The Streamlit dashboard should be launched and manually checked after installation, and the Docker stack should be built on the target machine before claiming deployment readiness.
