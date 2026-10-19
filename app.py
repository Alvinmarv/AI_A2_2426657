"""FastAPI service for Musanze Harvest Intelligence Service."""

import json
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

FEATURES = [
    "rainfall_mm",
    "avg_temp_c",
    "soil_moisture_pct",
    "fertilizer_kg",
    "field_size_ha",
    "season",
    "cooperative_support",
]

app = FastAPI(
    title="Musanze Harvest Intelligence Service",
    description=(
        "Predict crop yield, estimate high-risk status, "
        "and group farms using trained machine-learning models."
    ),
    version="1.0.0",
)


class FarmInput(BaseModel):
    rainfall_mm: float = Field(ge=0, le=5000)
    avg_temp_c: float = Field(ge=-20, le=60)
    soil_moisture_pct: float | None = Field(default=None, ge=0, le=100)
    fertilizer_kg: float | None = Field(default=None, ge=0, le=100000)
    field_size_ha: float = Field(gt=0, le=100000)
    season: Literal["A", "B", "C"]
    cooperative_support: int = Field(ge=0, le=1)


def load_model(filename):
    path = MODEL_DIR / filename
    if not path.exists():
        raise RuntimeError(
            f"Model not found: {path}. Run the training pipeline first."
        )
    return joblib.load(path)


@app.get("/")
def root():
    return {
        "service": "Musanze Harvest Intelligence Service",
        "verification_code": "AI-A2-6657",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    missing = [
        name
        for name in (
            "yield_regression.joblib",
            "high_risk_classifier.joblib",
            "farm_clusters.joblib",
        )
        if not (MODEL_DIR / name).exists()
    ]
    return {
        "status": "ok" if not missing else "models_missing",
        "models_missing": missing,
    }


@app.get("/model-info")
def model_info():
    metrics_path = BASE_DIR / "artifacts" / "metrics.json"
    selected_models = {}
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            selected_models = {
                name: metrics.get(name, {}).get("selected_model")
                for name in ("regression", "classification")
            }
            clustering = metrics.get("clustering", {})
            selected_models["clustering"] = {
                "algorithm": clustering.get("algorithm"),
                "n_clusters": clustering.get("n_clusters"),
            }
        except (OSError, ValueError):
            selected_models = {}

    model_files = (
        "yield_regression.joblib",
        "high_risk_classifier.joblib",
        "farm_clusters.joblib",
    )
    return {
        "verification_code": "AI-A2-6657",
        "selected_models": selected_models,
        "artifacts": {
            name: (MODEL_DIR / name).is_file() for name in model_files
        },
        "features": FEATURES,
    }


def input_frame(farm: FarmInput) -> pd.DataFrame:
    return pd.DataFrame([farm.model_dump()])[FEATURES]


@app.post("/predict/yield")
def predict_yield(farm: FarmInput):
    try:
        model = load_model("yield_regression.joblib")
        prediction = float(model.predict(input_frame(farm))[0])
        return {
            "predicted_yield_tons": round(max(0.0, prediction), 4),
            "model": "linear_regression",
        }
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/predict/risk")
def predict_risk(farm: FarmInput):
    try:
        model = load_model("high_risk_classifier.joblib")
        frame = input_frame(farm)
        prediction = int(model.predict(frame)[0])
        probability = None
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(frame)[0][1])

        return {
            "high_risk": prediction,
            "risk_label": "high" if prediction == 1 else "low",
            "high_risk_probability": (
                round(probability, 4) if probability is not None else None
            ),
            "model": "random_forest",
        }
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/predict/cluster")
def predict_cluster(farm: FarmInput):
    try:
        bundle = load_model("farm_clusters.joblib")
        frame = input_frame(farm)
        transformed = bundle["preprocessor"].transform(frame)
        cluster = int(bundle["model"].predict(transformed)[0])
        return {"cluster": cluster, "algorithm": "KMeans"}
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
