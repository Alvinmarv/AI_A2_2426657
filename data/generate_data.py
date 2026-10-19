"""Generate the reproducible synthetic dataset for AI-A2-6657."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


REGISTRATION_DIGITS = "2426657"
LAST_FOUR_DIGITS = 6657
VERIFICATION_CODE = "AI-A2-6657"

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"


def generate_dataset(seed: int = LAST_FOUR_DIGITS) -> pd.DataFrame:
    """Generate base farm records, deliberate missing values and duplicates."""
    if seed != LAST_FOUR_DIGITS:
        raise ValueError(
            f"Seed must equal the last four registration digits: "
            f"{LAST_FOUR_DIGITS}."
        )

    rng = np.random.default_rng(seed)
    n_rows = 240 + (seed % 41)

    rainfall = rng.uniform(600, 1400, n_rows)
    temperature = rng.uniform(12, 26, n_rows)
    moisture = rng.uniform(20, 80, n_rows)
    fertilizer = rng.uniform(0, 250, n_rows)
    field_size = rng.uniform(0.2, 4.0, n_rows)
    season = rng.choice(["A", "B"], n_rows)
    support = rng.integers(0, 2, n_rows)

    season_effect = np.where(season == "A", 0.50, -0.20)
    noise = rng.normal(0, 0.80, n_rows)

    yield_tons = (
        0.004 * rainfall
        + 0.06 * moisture
        + 0.02 * fertilizer
        + 1.30 * field_size
        - 0.25 * np.abs(temperature - 18)
        + 0.70 * support
        + season_effect
        + noise
    )
    yield_tons = np.clip(yield_tons, 0.50, None)

    df = pd.DataFrame({
        "rainfall_mm": rainfall,
        "avg_temp_c": temperature,
        "soil_moisture_pct": moisture,
        "fertilizer_kg": fertilizer,
        "field_size_ha": field_size,
        "season": season,
        "cooperative_support": support,
        "yield_tons": yield_tons,
    })

    # High risk means yield per hectare is below the base-data median.
    yield_per_hectare = df["yield_tons"] / df["field_size_ha"]
    df["high_risk"] = (
        yield_per_hectare < yield_per_hectare.median()
    ).astype(int)

    # Deliberately introduce missing values using the same random generator.
    missing_count = round(0.05 * n_rows)
    moisture_rows = rng.choice(n_rows, size=missing_count, replace=False)
    fertilizer_rows = rng.choice(n_rows, size=missing_count, replace=False)

    df.loc[moisture_rows, "soil_moisture_pct"] = np.nan
    df.loc[fertilizer_rows, "fertilizer_kg"] = np.nan

    # Append three exact duplicate rows after introducing missing values.
    df = pd.concat([df, df.iloc[:3]], ignore_index=True)

    return df


def write_dataset(seed: int = LAST_FOUR_DIGITS):
    """Generate the personalized CSV and its stable fingerprint record."""
    if seed != LAST_FOUR_DIGITS:
        raise ValueError(
            f"Seed must be {LAST_FOUR_DIGITS} for this registration number."
        )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = generate_dataset(seed)
    csv_path = DATA_DIR / f"AI_A2_{REGISTRATION_DIGITS}.csv"
    df.to_csv(csv_path, index=False, float_format="%.4f")

    fingerprint = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    fingerprint_path = ARTIFACTS_DIR / "data_fingerprint.txt"
    fingerprint_path.write_text(
        f"Student registration: 24/26657\n"
        f"Verification code: {VERIFICATION_CODE}\n"
        f"Seed: {seed}\n"
        f"Base rows: {240 + (seed % 41)}\n"
        f"Rows including appended duplicates: {len(df)}\n"
        f"CSV SHA-256: {fingerprint}\n",
        encoding="utf-8",
    )

    print("Musanze Harvest Intelligence Service")
    print(f"Verification code: {VERIFICATION_CODE}")
    print(f"Seed: {seed}")
    print(f"Base rows: {240 + (seed % 41)}")
    print(f"Rows after duplicates: {len(df)}")
    print(f"Missing soil values: {df['soil_moisture_pct'].isna().sum()}")
    print(f"Missing fertilizer values: {df['fertilizer_kg'].isna().sum()}")
    print(f"Exact duplicate rows: {df.duplicated().sum()}")
    print(f"Dataset: {csv_path}")
    print(f"SHA-256: {fingerprint}")
    return csv_path, fingerprint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the personalized AI assignment dataset."
    )
    parser.add_argument("--seed", type=int, default=LAST_FOUR_DIGITS)
    args = parser.parse_args()
    try:
        write_dataset(args.seed)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
