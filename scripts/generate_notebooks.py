"""Create two reproducible EDA notebooks with 15 plots each."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"
OUT.mkdir(exist_ok=True)

common = '''from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA = Path("data/raw")
if not DATA.exists():
    DATA = Path("../data/raw")
plt.rcParams.update({"figure.figsize": (8, 4), "axes.grid": True, "figure.dpi": 110})
def finish(title, xlabel=None, ylabel=None):
    plt.title(title)
    if xlabel: plt.xlabel(xlabel)
    if ylabel: plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()
    plt.close()
'''

material_code = common + '''
df = pd.read_csv(DATA / "DS1_material_properties_5500.csv")
print(f"Rows: {len(df):,}; columns: {len(df.columns)}")
print(df.isna().mean().sort_values(ascending=False).head(10))

# 1. Crystal-system distribution
 df = df
'''.replace('\n df = df\n','\n')
# Append clean individual charts; each numbered chart is its own Matplotlib figure.
material_plots = [
'''df["crystal_system"].value_counts().plot(kind="bar"); finish("1. Materials by crystal system", "Crystal system", "Count")''',
'''df["category"].value_counts().head(15).plot(kind="bar"); finish("2. Most common material categories", "Category", "Count")''',
'''df["formation_energy_per_atom_eV"].hist(bins=40); finish("3. Formation energy distribution", "eV/atom", "Count")''',
'''df["band_gap_eV"].hist(bins=40); finish("4. Band gap distribution", "eV", "Count")''',
'''df["bulk_modulus_GPa"].hist(bins=40); finish("5. Bulk modulus distribution", "GPa", "Count")''',
'''df["shear_modulus_GPa"].hist(bins=40); finish("6. Shear modulus distribution", "GPa", "Count")''',
'''df["poisson_ratio"].hist(bins=30); finish("7. Poisson ratio distribution", "Poisson ratio", "Count")''',
'''df["density_g_cm3"].hist(bins=40); finish("8. Density distribution", "g/cm³", "Count")''',
'''df["melting_point_K"].hist(bins=40); finish("9. Melting point distribution", "Kelvin", "Count")''',
'''df["n_elements"].value_counts().sort_index().plot(kind="bar"); finish("10. Number of elements per formula", "Element count", "Materials")''',
'''df["spacegroup_number"].hist(bins=46); finish("11. Space-group number distribution", "Space-group number", "Count")''',
'''corr = df[["n_elements", "spacegroup_number", "formation_energy_per_atom_eV", "band_gap_eV", "bulk_modulus_GPa", "shear_modulus_GPa", "poisson_ratio", "density_g_cm3", "nsites", "volume_A3", "melting_point_K"]].corr(numeric_only=True); plt.imshow(corr, aspect="auto"); plt.xticks(range(len(corr)), corr.columns, rotation=90); plt.yticks(range(len(corr)), corr.columns); plt.colorbar(label="Correlation"); finish("12. Numeric correlation matrix")''',
'''groups = [g["band_gap_eV"].dropna().values for _, g in df.groupby("crystal_system")]; labels = list(df.groupby("crystal_system").groups.keys()); plt.boxplot(groups, tick_labels=labels, showfliers=False); plt.xticks(rotation=30); finish("13. Band gap by crystal system", "Crystal system", "Band gap (eV)")''',
'''top = df.groupby("category")["density_g_cm3"].median().sort_values(ascending=False).head(10); top.sort_values().plot(kind="barh"); finish("14. Median density for 10 categories", "Median density (g/cm³)", "Category")''',
'''plt.scatter(df["bulk_modulus_GPa"], df["shear_modulus_GPa"], s=8, alpha=.35); finish("15. Bulk vs shear modulus", "Bulk modulus (GPa)", "Shear modulus (GPa)")''',
'''df.boxplot(column="formation_energy_per_atom_eV", by="is_stable", grid=True); plt.suptitle(""); finish("16. Formation energy by supplied stability flag", "is_stable (0/1)", "eV/atom")''',
]

financial_code = common + '''
prices = pd.read_csv(DATA / "DS2_commodity_prices_10yr.csv", parse_dates=["date"])
cross = pd.read_csv(DATA / "DS4_crossdomain_features_daily.csv", parse_dates=["date"])
df = prices.merge(cross, on=["date", "commodity"], how="left", validate="one_to_one").sort_values(["commodity", "date"])
df["next_day_return"] = df.groupby("commodity")["daily_return"].shift(-1)
print(f"Rows: {len(df):,}; commodities: {df['commodity'].nunique()}; dates: {df['date'].min().date()} to {df['date'].max().date()}")

# 1. Observation counts per commodity
'''
financial_plots = [
'''df["commodity"].value_counts().plot(kind="bar"); finish("1. Observations by commodity", "Commodity", "Rows")''',
'''g = df[df["commodity"] == "Steel_HRC"].sort_values("date"); plt.plot(g["date"], g["close"]); finish("2. Steel HRC price history", "Date", "Close price")''',
'''pivot = df.pivot(index="date", columns="commodity", values="close"); normalized = pivot / pivot.bfill().iloc[0] * 100; normalized.plot(ax=plt.gca(), alpha=.8); finish("3. Normalized commodity price paths", "Date", "Index (first common price = 100)")''',
'''df["daily_return"].dropna().hist(bins=100); finish("4. Daily-return distribution", "Daily log return", "Count")''',
'''df.boxplot(column="volatility_21d_ann", by="commodity", grid=True, showfliers=False); plt.xticks(rotation=30); plt.suptitle(""); finish("5. 21-day annualized volatility by commodity", "Commodity", "Volatility")''',
'''np.log1p(df["volume"].clip(lower=0)).hist(bins=50); finish("6. Log-transformed trading volume", "log(1 + volume)", "Count")''',
'''df["rsi_14"].dropna().hist(bins=40); finish("7. RSI-14 distribution", "RSI", "Count")''',
'''df["bollinger_z"].dropna().hist(bins=60); finish("8. Bollinger z-score distribution", "Bollinger z", "Count")''',
'''ret = df.dropna(subset=["return_21d"]); ret.boxplot(column="return_21d", by="commodity", grid=True, showfliers=False); plt.xticks(rotation=30); plt.suptitle(""); finish("9. 21-day return by commodity", "Commodity", "21-day return")''',
'''g = df[df["commodity"] == "Steel_HRC"].sort_values("date"); plt.plot(g["date"], g["mqi"]); finish("10. Steel HRC Material Quality Index", "Date", "MQI")''',
'''df.groupby("commodity")["supply_disruption_prob"].mean().sort_values().plot(kind="barh"); finish("11. Mean supply disruption probability", "Mean probability", "Commodity")''',
'''df.groupby("commodity")["substitution_elasticity"].mean().sort_values().plot(kind="barh"); finish("12. Mean substitution elasticity", "Mean elasticity", "Commodity")''',
'''df.groupby("commodity")["green_premium_per_kg"].mean().sort_values().plot(kind="barh"); finish("13. Mean green premium", "USD/kg", "Commodity")''',
'''cols = ["daily_return", "return_5d", "return_21d", "volatility_21d_ann", "rsi_14", "bollinger_z", "mqi", "mqi_21d_trend", "supply_disruption_prob", "substitution_elasticity"]; c = df[cols].corr(); plt.imshow(c, aspect="auto", vmin=-1, vmax=1); plt.xticks(range(len(cols)), cols, rotation=90); plt.yticks(range(len(cols)), cols); plt.colorbar(label="Correlation"); finish("14. Technical and cross-domain correlations")''',
'''sample = df.dropna(subset=["mqi_21d_trend", "next_day_return"]).sample(min(4000, df["mqi_21d_trend"].notna().sum()), random_state=42); plt.scatter(sample["mqi_21d_trend"], sample["next_day_return"], alpha=.25, s=8); finish("15. MQI trend vs next-day return", "21-day MQI trend", "Next-day log return")''',
'''df.boxplot(column="term_spread", by="commodity", grid=True, showfliers=False); plt.xticks(rotation=30); plt.suptitle(""); finish("16. Term spread by commodity", "Commodity", "Term spread")''',
]

def notebook(title, markdown, setup, plot_cells):
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(f"# {title}\n\n{markdown}\n\n**Data note:** these are supplied internship sample datasets. Figures describe the sample and should not be presented as verified live data."),
        nbf.v4.new_code_cell(setup),
    ] + [nbf.v4.new_code_cell(plot) for plot in plot_cells]
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
    return nb

nbf.write(notebook("Material Dataset EDA", "Exploratory analysis of composition, structure and property fields in DS-1. Each numbered chart is a separate figure.", material_code, material_plots), OUT / "01_material_eda.ipynb")
nbf.write(notebook("Financial and Cross-Domain EDA", "Exploratory analysis of commodity technical indicators and DS-4 cross-domain signals, with a same-date merge.", financial_code, financial_plots), OUT / "02_financial_eda.ipynb")
print("Generated", OUT / "01_material_eda.ipynb", "and", OUT / "02_financial_eda.ipynb")
