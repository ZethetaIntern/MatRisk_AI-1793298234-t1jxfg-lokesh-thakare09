# MatRisk AI - Submission Readiness Checklist

Prepared as a development status note for the current prototype. The internship brief contains a much larger research and engineering scope, so this file intentionally distinguishes finished prototype components from components that still require full implementation.

## Implemented in this prototype

- Loads and validates the six supplied CSV datasets, preserving them as assignment sample data.
- Runs material-data quality checks for missing values, duplicates, basic physical bounds, and elastic-relationship consistency.
- Trains a tabular ExtraTrees baseline for formation energy, band gap, bulk modulus, and shear modulus, using a group-aware holdout.
- Runs a separate commodity next-day-return experiment with a chronological holdout and four feature ablations.
- Trains a bridge structural-deficiency classifier and converts its output into a clearly labelled illustrative PD/LGD/EAD-style loss proxy.
- Runs five assumption-driven Monte Carlo material-financial stress scenarios with 10,000 paths each.
- Calculates a material-substitution emissions and cost comparison using the supplied sample data.
- Screens existing materials against simple requested-property and cost criteria. This is a screening heuristic, not a generative alloy design model.
- Includes a Streamlit dashboard source file with seven sections, a small FastAPI wrapper, saved model files, plots, metric summaries, CI configuration, Docker configuration, and a technical report.
- Includes two EDA notebooks. Both were executed successfully in the build environment and contain 16 chart outputs apiece.
- The automated Python test suite passed 19 tests with 97% statement coverage in `src/matrisk` for the recorded local run.

## Still required for the full original brief

- A full crystal-graph neural network (CGNN) that constructs real atomic graphs from crystallographic coordinates.
- A trained physics-informed neural network (PINN) whose physics losses are part of model training. The current physics audit is not a PINN.
- 100,000+ real material records acquired from Materials Project/AFLOW and verified API ingestion.
- The brief's 10+ year / multi-commodity live or sourced financial-data workflow, provenance checks, and reproducible data versioning with a configured DVC remote.
- Real National Bridge Inventory data and a validated survival-analysis model (for example, DeepSurv) with a reported C-index. The current bridge classifier is not survival analysis and its output is not a calibrated loan-default probability.
- A trained WGAN-GP inverse material generator. The current screen ranks existing sample records only.
- A fully validated commodity strategy meeting the stated Sharpe threshold. Current out-of-sample toy backtest results do not meet the target.
- Complete SHAP and partial-dependence explanations across all models, sensitivity-analysis sweeps, and comprehensive uncertainty/calibration analysis.
- A production-grade insurance catastrophe model and externally calibrated loss-severity assumptions.
- Actual Streamlit browser execution/visual QA, Docker build and `docker compose up` verification, and end-to-end API deployment on the target machine. Source/configuration is included but was not verified in this environment.
- Configured DVC and MLflow remotes, the full data acquisition pipeline, and 100% public API documentation.
- Check the internship portal and supervisor's instructions for final repository transfer and institutional/IP requirements before submitting or transferring any repository.

## Important interpretation notes

- The six CSV files are the provided sample data, not proof of a live production dataset or an independently verified official data feed.
- Stress-scenario impact percentages, loss distributions and the bridge loss conversion are illustrative assumptions for a prototype. Do not present them as calibrated institutional risk estimates.
- The unusually high bridge classifier scores should trigger a target/feature leakage and sample-generation audit before any real-world claim.
- Commodity directional accuracy is close to 50%, and none of the reported toy strategies reaches the assignment's Sharpe target. This is a valid negative finding and should not be hidden.
- In the report, label all results as sample-based and state the limitations. Add your own observations after you run the project locally.
