"""Run farm predictions from the command line."""

import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from app import FarmInput, input_frame, load_model


def main():
    parser = argparse.ArgumentParser(
        description="Predict farm yield, risk, and cluster."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="Optional JSON file containing farm information.",
    )
    args = parser.parse_args()

    sample = {
        "rainfall_mm": 870.6,
        "avg_temp_c": 20.0,
        "soil_moisture_pct": 40.0,
        "fertilizer_kg": 120.0,
        "field_size_ha": 1.5,
        "season": "A",
        "cooperative_support": 1,
    }

    try:
        if args.input:
            with args.input.open("r", encoding="utf-8") as file:
                sample = json.load(file)

        farm = FarmInput.model_validate(sample)
        frame = input_frame(farm)

        yield_model = load_model("yield_regression.joblib")
        risk_model = load_model("high_risk_classifier.joblib")
        cluster_bundle = load_model("farm_clusters.joblib")

        predicted_yield = max(
            0.0, float(yield_model.predict(frame)[0])
        )

        risk_prediction = int(risk_model.predict(frame)[0])
        risk_probability = float(risk_model.predict_proba(frame)[0][1])

        transformed = cluster_bundle["preprocessor"].transform(frame)
        cluster = int(cluster_bundle["model"].predict(transformed)[0])

        results = {
            "project": "Musanze Harvest Intelligence Service",
            "verification_code": "AI-A2-6657",
            "predicted_yield_tons": round(predicted_yield, 4),
            "risk_label": "high" if risk_prediction == 1 else "low",
            "high_risk_probability": round(risk_probability, 4),
            "farm_cluster": cluster,
        }

        print(json.dumps(results, indent=2))

    except (OSError, ValueError, RuntimeError, ValidationError) as error:
        print(f"Prediction failed: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
