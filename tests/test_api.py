import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import app
from data.generate_data import generate_dataset
from src.pipeline import validate_dataset

client = TestClient(app)

VALID_FARM = {
    "rainfall_mm": 870.6,
    "avg_temp_c": 20.0,
    "soil_moisture_pct": 40.0,
    "fertilizer_kg": 120.0,
    "field_size_ha": 1.5,
    "season": "A",
    "cooperative_support": 1,
}


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info_endpoint():
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["verification_code"] == "AI-A2-6657"
    assert body["selected_models"]["regression"] == "linear_regression"
    assert body["selected_models"]["clustering"]["n_clusters"] == 2
    assert all(body["artifacts"].values())


def test_yield_prediction():
    response = client.post("/predict/yield", json=VALID_FARM)
    assert response.status_code == 200
    assert response.json()["predicted_yield_tons"] >= 0


def test_risk_prediction():
    response = client.post("/predict/risk", json=VALID_FARM)
    assert response.status_code == 200
    assert response.json()["risk_label"] in ["high", "low"]
    assert 0 <= response.json()["high_risk_probability"] <= 1


def test_cluster_prediction():
    response = client.post("/predict/cluster", json=VALID_FARM)
    assert response.status_code == 200
    assert isinstance(response.json()["cluster"], int)


def test_invalid_farm_input():
    invalid_farm = VALID_FARM.copy()
    invalid_farm["rainfall_mm"] = -10
    response = client.post("/predict/yield", json=invalid_farm)
    assert response.status_code == 422


def test_invalid_season_is_rejected():
    invalid_farm = VALID_FARM.copy()
    invalid_farm["season"] = "D"
    response = client.post("/predict/yield", json=invalid_farm)
    assert response.status_code == 422


def test_missing_model_returns_service_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr("app.MODEL_DIR", tmp_path)
    response = client.post("/predict/yield", json=VALID_FARM)
    assert response.status_code == 503
    assert "Run the training pipeline first" in response.json()["detail"]


def test_generated_data_is_deterministic_and_includes_injected_quality_issues():
    first = generate_dataset()
    second = generate_dataset()
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 258
    assert first.duplicated().sum() == 3
    assert first["soil_moisture_pct"].isna().sum() > 0
    assert first["fertilizer_kg"].isna().sum() > 0


def test_dataset_validator_accepts_generated_data():
    validate_dataset(generate_dataset())


def test_dataset_validator_rejects_out_of_range_values():
    invalid_data = generate_dataset()
    invalid_data.loc[0, "rainfall_mm"] = 5001
    with pytest.raises(ValueError, match="outside valid ranges"):
        validate_dataset(invalid_data)


def test_dataset_validator_rejects_missing_required_column():
    incomplete_data = generate_dataset().drop(columns=["field_size_ha"])
    with pytest.raises(ValueError, match="Missing dataset columns"):
        validate_dataset(incomplete_data)


def test_dataset_validator_rejects_zero_field_size():
    invalid_data = generate_dataset()
    invalid_data.loc[0, "field_size_ha"] = 0
    with pytest.raises(ValueError, match="outside valid ranges"):
        validate_dataset(invalid_data)
