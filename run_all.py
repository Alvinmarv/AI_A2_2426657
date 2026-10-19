"""Regenerate data, train models, and run automated tests."""

import argparse
import json
import sys
from pathlib import Path

import pytest

from data.generate_data import (
    LAST_FOUR_DIGITS,
    REGISTRATION_DIGITS,
    write_dataset,
)
from src.pipeline import train_and_evaluate

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(
        description="Recreate the personalized dataset, models, and evidence."
    )
    parser.add_argument("--student-id", default="24/26657")
    parser.add_argument("--seed", type=int, default=LAST_FOUR_DIGITS)
    args = parser.parse_args()

    student_digits = "".join(
        character for character in args.student_id if character.isdigit()
    )
    if student_digits != REGISTRATION_DIGITS:
        parser.error(f"--student-id must be 24/26657 for this project.")
    if args.seed != LAST_FOUR_DIGITS:
        parser.error(f"--seed must be {LAST_FOUR_DIGITS} for this project.")

    print("=" * 60)
    print("MUSANZE HARVEST INTELLIGENCE SERVICE")
    print(f"Student: {args.student_id} | Seed: {args.seed}")
    print("=" * 60)

    print("\n[1/3] Regenerating dataset and fingerprint...")
    csv_path, fingerprint = write_dataset(args.seed)

    print("\n[2/3] Training and evaluating models...")
    try:
        metrics = train_and_evaluate()
    except Exception as error:
        print(f"Training pipeline failed: {error}")
        return 1

    print("\n[3/3] Running automated tests...")
    result = pytest.main(["-q"])

    artifacts = [
        str(csv_path.relative_to(ROOT)),
        "artifacts/data_fingerprint.txt",
        "artifacts/metrics.json",
        "artifacts/regression_metrics.json",
        "artifacts/classification_metrics.json",
        "artifacts/clustering_metrics.json",
        "artifacts/farm_cluster_assignments.csv",
        "artifacts/plots/yield_actual_vs_predicted.png",
        "artifacts/plots/classification_confusion_matrix.png",
        "artifacts/plots/farm_clusters.png",
        "models/yield_regression.joblib",
        "models/high_risk_classifier.joblib",
        "models/farm_clusters.joblib",
    ]
    manifest = {
        "student_id": args.student_id,
        "registration_digits": student_digits,
        "seed": args.seed,
        "verification_code": f"AI-A2-{args.seed:04d}",
        "rows_after_cleaning": metrics["cleaning"]["rows_after_cleaning"],
        "sha256": fingerprint,
        "artifacts": artifacts,
        "tests_passed": result == pytest.ExitCode.OK,
    }
    (ROOT / "artifacts" / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("\nGenerated artifacts:")
    for artifact in artifacts:
        print(f"- {artifact}")
    print("- artifacts/run_manifest.json")
    print(f"\nDataset SHA-256: {fingerprint}")

    if result != pytest.ExitCode.OK:
        print("\nSome automated tests failed.")
        return int(result)

    print("\nSUCCESS: Pipeline completed and all tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
