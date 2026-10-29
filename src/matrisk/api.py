"""Small FastAPI wrapper for selected MatRisk AI prototype outputs."""
from __future__ import annotations
from pathlib import Path
import sys
import json
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from matrisk.data import load_all  # noqa: E402
from matrisk.material_model import predict_material  # noqa: E402
from matrisk.stress import SCENARIOS, run_stress_test  # noqa: E402

app = FastAPI(title="MatRisk AI Prototype API", version="0.1.0", description="Educational prototype; not a production risk service.")

class MaterialRequest(BaseModel):
    material_id: str = Field(..., description="ID from DS1 material sample")

class StressRequest(BaseModel):
    scenario: str
    exposure_usd: float = Field(default=10_000_000, ge=0)
    severity_multiplier: float = Field(default=1.0, ge=0.5, le=2.0)
    n_paths: int = Field(default=10_000, ge=100, le=100_000)

@app.get("/health")
def health():
    return {"status": "ok", "service": "matrisk-ai-prototype", "models_dir_exists": (ROOT / "models").exists()}

@app.get("/material/{material_id}")
def material(material_id: str):
    model_path = ROOT / "models" / "material_extratrees.joblib"
    if not model_path.exists():
        raise HTTPException(status_code=503, detail="Model missing. Run python -m matrisk.train_all first.")
    data = load_all()
    rows = data["materials"].loc[data["materials"]["material_id"].astype(str) == material_id]
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"Material ID {material_id} not found in DS1 sample")
    model = joblib.load(model_path)
    record = rows.iloc[0]
    return {"material_id": material_id, "formula": record["formula"], "predictions": predict_material(model, record)}

@app.post("/stress")
def stress(payload: StressRequest):
    if payload.scenario not in SCENARIOS:
        raise HTTPException(status_code=400, detail=f"Scenario must be one of: {list(SCENARIOS)}")
    result = run_stress_test(payload.exposure_usd, payload.scenario, payload.severity_multiplier, payload.n_paths)
    result.pop("samples", None)
    return result
