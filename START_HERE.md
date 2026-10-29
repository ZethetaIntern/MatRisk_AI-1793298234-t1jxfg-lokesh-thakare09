# Start Here - MatRisk AI

Welcome! This project links material condition to financial risk. Start by exploring the sample data and baseline results before trying to run every component.

## What to open first

1. `README.md` - project overview and setup steps.
2. `reports/technical_report.pdf` - formatted report with the methods, measured results, limitations, charts and references.
3. `reports/viva_notes.md` - plain-language explanations for presenting the project and answering questions.
4. `reports/submission_readiness.md` - what is implemented and tested, and which items from the original brief remain incomplete.
5. `notebooks/01_material_eda.ipynb` and `notebooks/02_financial_eda.ipynb` - two executed EDA notebooks containing 16 charts each.

## Run it on Windows

Install Python 3.11 or newer. In PowerShell, open this folder and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
$env:PYTHONPATH = "src"
python -m matrisk.train_all
streamlit run app.py
```

Then visit `http://localhost:8501`. For the optional API, open a second terminal in the activated environment and run:

```powershell
$env:PYTHONPATH = "src"
uvicorn matrisk.api:app --reload --port 8000
```

## What you can honestly say in a review

This is a sample-data prototype. It includes tabular baselines and a Streamlit app source, but it does not yet implement every specialist architecture in the internship brief. In particular, the material model is not a crystal-graph neural network, the physics audit is not a trained PINN, the bridge model is not a survival model, and inverse material design is a screening heuristic rather than a WGAN-GP generator. The report records actual prototype metrics rather than claiming the brief's stretch targets have been met.

Before you present it, run the commands on your own machine, inspect the metrics, click through the dashboard, and make at least one change you can explain. Follow your internship's AI-assistance and repository-transfer policies.
