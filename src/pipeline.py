"""Machine-learning pipeline for the farm dataset."""

import json
import hashlib
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    silhouette_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "AI_A2_2426657.csv"
if not DATA_PATH.exists():
    DATA_PATH = ROOT / "AI_A2_2426657" / "data" / "AI_A2_2426657.csv"
MODEL_DIR = ROOT / "models"
ARTIFACTS = ROOT / "artifacts"
PLOTS = ARTIFACTS / "plots"

FEATURES = [
    "rainfall_mm",
    "avg_temp_c",
    "soil_moisture_pct",
    "fertilizer_kg",
    "field_size_ha",
    "season",
    "cooperative_support",
]


def make_preprocessor(frame, scale=False):
    """Impute missing data and encode categorical columns."""
    numeric = frame.select_dtypes(include=["number"]).columns.tolist()
    categorical = [c for c in frame.columns if c not in numeric]

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]

    return ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), numeric),
        ("categorical", Pipeline(categorical_steps), categorical),
    ])


def validate_dataset(frame):
    """Validate required columns, missing-value policy, and feature ranges."""
    required = set(FEATURES + ["yield_tons", "high_risk"])
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"Missing dataset columns: {sorted(missing_columns)}"
        )
    if frame.empty:
        raise ValueError("Dataset must contain at least one row.")

    allowed_missing = {"soil_moisture_pct", "fertilizer_kg"}
    for column in required - allowed_missing:
        if frame[column].isna().any():
            raise ValueError(f"Unexpected missing values in {column}.")

    numeric_ranges = {
        "rainfall_mm": (0, 5000),
        "avg_temp_c": (-20, 60),
        "soil_moisture_pct": (0, 100),
        "fertilizer_kg": (0, 100000),
        "field_size_ha": (0, 100000),
        "yield_tons": (0, float("inf")),
    }
    for column, (lower, upper) in numeric_ranges.items():
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise ValueError(f"Dataset column {column} must be numeric.")
        values = frame[column].dropna()
        lower_bound_ok = (
            values.gt(lower).all()
            if column == "field_size_ha"
            else values.ge(lower).all()
        )
        if not np.isfinite(values).all() or not values.between(
            lower, upper
        ).all() or not lower_bound_ok:
            raise ValueError(f"Values in {column} are outside valid ranges.")

    if not frame["season"].isin(["A", "B", "C"]).all():
        raise ValueError("Season values must be A, B, or C.")
    if not frame["cooperative_support"].isin([0, 1]).all():
        raise ValueError("cooperative_support values must be 0 or 1.")
    if not frame["high_risk"].isin([0, 1]).all():
        raise ValueError("high_risk values must be 0 or 1.")


def load_and_clean_data():
    """Load the CSV and remove exact duplicate rows."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    original = pd.read_csv(DATA_PATH)

    validate_dataset(original)

    report = {
        "rows_before_cleaning": int(len(original)),
        "duplicate_rows_removed": int(original.duplicated().sum()),
        "missing_values_before_cleaning": {
            k: int(v)
            for k, v in original.isna().sum().items()
            if v > 0
        },
        "sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
    }

    cleaned = original.drop_duplicates().reset_index(drop=True)
    report["rows_after_cleaning"] = int(len(cleaned))
    return cleaned, report


def train_and_evaluate():
    """Train models, evaluate them, and save artifacts."""
    for folder in (MODEL_DIR, ARTIFACTS, PLOTS):
        folder.mkdir(parents=True, exist_ok=True)

    df, cleaning = load_and_clean_data()

    results = {
        "project": "Musanze Harvest Intelligence Service",
        "registration": "24/26657",
        "verification_code": "AI-A2-6657",
        "cleaning": cleaning,
        "regression": {},
        "classification": {},
        "clustering": {},
    }

    # Regression: estimate crop yield.
    X = df[FEATURES]
    y = df["yield_tons"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=6657
    )

    regressors = {
        "mean_baseline": Pipeline([
            ("prep", make_preprocessor(X_train)),
            ("model", DummyRegressor(strategy="mean")),
        ]),
        "linear_regression": Pipeline([
            ("prep", make_preprocessor(X_train, scale=True)),
            ("model", LinearRegression()),
        ]),
        "random_forest": Pipeline([
            ("prep", make_preprocessor(X_train)),
            ("model", RandomForestRegressor(
                n_estimators=150, random_state=6657
            )),
        ]),
    }

    best_regressor = None
    best_regressor_name = None
    best_rmse = float("inf")

    for name, model in regressors.items():
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, prediction)))
        results["regression"][name] = {
            "mae": float(mean_absolute_error(y_test, prediction)),
            "rmse": rmse,
            "r2": float(r2_score(y_test, prediction)),
        }

        if rmse < best_rmse:
            best_rmse = rmse
            best_regressor = model
            best_regressor_name = name

    joblib.dump(best_regressor, MODEL_DIR / "yield_regression.joblib")
    results["regression"]["selected_model"] = best_regressor_name
    results["regression"]["selection_criterion"] = (
        "Lowest test RMSE across the mean baseline and two candidates."
    )

    # Plot actual versus predicted yields.
    prediction = best_regressor.predict(X_test)
    residuals = np.asarray(y_test) - np.asarray(prediction)
    results["regression"]["residual_analysis"] = {
        "definition": "actual minus predicted yield, in tons",
        "mean_bias": float(np.mean(residuals)),
        "standard_deviation": float(np.std(residuals)),
        "maximum_absolute_error": float(np.max(np.abs(residuals))),
        "interpretation": (
            "Positive mean bias indicates underprediction; negative mean "
            "bias indicates overprediction."
        ),
    }
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_test, prediction)
    ax.set_xlabel("Actual yield (tons)")
    ax.set_ylabel("Predicted yield (tons)")
    ax.set_title(f"Yield prediction: {best_regressor_name}")
    fig.tight_layout()
    fig.savefig(PLOTS / "yield_actual_vs_predicted.png")
    plt.close(fig)

    # Classification: estimate high-risk status.
    # yield_tons is excluded because the target is derived from yield per hectare.
    Xc = df[FEATURES]
    yc = df["high_risk"].astype(int)

    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        Xc, yc, test_size=0.2, random_state=6657, stratify=yc
    )

    classifiers = {
        "majority_baseline": Pipeline([
            ("prep", make_preprocessor(Xc_train)),
            ("model", DummyClassifier(
                strategy="most_frequent", random_state=6657
            )),
        ]),
        "logistic_regression": Pipeline([
            ("prep", make_preprocessor(Xc_train, scale=True)),
            ("model", LogisticRegression(max_iter=2000)),
        ]),
        "random_forest": Pipeline([
            ("prep", make_preprocessor(Xc_train)),
            ("model", RandomForestClassifier(
                n_estimators=150, random_state=6657,
                class_weight="balanced"
            )),
        ]),
    }

    best_classifier = None
    best_classifier_name = None
    best_f1 = -1

    for name, model in classifiers.items():
        model.fit(Xc_train, yc_train)
        prediction = model.predict(Xc_test)
        probability = model.predict_proba(Xc_test)[:, 1]
        matrix = confusion_matrix(yc_test, prediction, labels=[0, 1])

        score = float(f1_score(yc_test, prediction, zero_division=0))
        results["classification"][name] = {
            "accuracy": float(accuracy_score(yc_test, prediction)),
            "precision": float(precision_score(
                yc_test, prediction, zero_division=0
            )),
            "recall": float(recall_score(
                yc_test, prediction, zero_division=0
            )),
            "f1": score,
            "roc_auc": float(roc_auc_score(yc_test, probability)),
            "confusion_matrix": matrix.tolist(),
            "error_counts": {
                "true_negative": int(matrix[0, 0]),
                "false_positive": int(matrix[0, 1]),
                "false_negative": int(matrix[1, 0]),
                "true_positive": int(matrix[1, 1]),
            },
        }

        if score > best_f1:
            best_f1 = score
            best_classifier = model
            best_classifier_name = name

    joblib.dump(
        best_classifier, MODEL_DIR / "high_risk_classifier.joblib"
    )
    results["classification"]["selected_model"] = best_classifier_name
    results["classification"]["selection_criterion"] = (
        "Highest test F1 across the majority baseline and two candidates."
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(
        best_classifier, Xc_test, yc_test, ax=ax, colorbar=False
    )
    ax.set_title("High-risk classification")
    fig.tight_layout()
    fig.savefig(PLOTS / "classification_confusion_matrix.png")
    plt.close(fig)

    # Clustering: group farms using input features only.
    cluster_frame = df[FEATURES]
    cluster_preprocessor = make_preprocessor(
        cluster_frame, scale=True
    )
    transformed = cluster_preprocessor.fit_transform(cluster_frame)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    cluster_comparison = {}
    cluster_candidates = {}
    for cluster_count in range(2, 7):
        candidate = KMeans(
            n_clusters=cluster_count, random_state=6657, n_init=10
        )
        candidate_labels = candidate.fit_predict(transformed)
        score = float(silhouette_score(transformed, candidate_labels))
        cluster_comparison[str(cluster_count)] = {
            "silhouette_score": score,
            "inertia": float(candidate.inertia_),
        }
        cluster_candidates[cluster_count] = (candidate, candidate_labels)

    selected_cluster_count = max(
        cluster_candidates,
        key=lambda count: cluster_comparison[str(count)]["silhouette_score"],
    )
    cluster_model, labels = cluster_candidates[selected_cluster_count]

    joblib.dump({
        "preprocessor": cluster_preprocessor,
        "model": cluster_model,
        "features": FEATURES,
    }, MODEL_DIR / "farm_clusters.joblib")

    cluster_profiles = {}
    for cluster_id in range(selected_cluster_count):
        members = cluster_frame.iloc[np.flatnonzero(labels == cluster_id)]
        cluster_profiles[str(cluster_id)] = {
            "size": int(len(members)),
            "numeric_feature_means": {
                column: float(members[column].mean())
                for column in cluster_frame.select_dtypes(
                    include=["number"]
                ).columns
            },
            "season_share": {
                str(season): float(share)
                for season, share in members["season"].value_counts(
                    normalize=True
                ).items()
            },
        }

    results["clustering"] = {
        "algorithm": "KMeans",
        "n_clusters": selected_cluster_count,
        "selection_criterion": (
            "Highest silhouette score among K=2 through K=6."
        ),
        "comparison": cluster_comparison,
        "cluster_sizes": {
            str(int(k)): int(v)
            for k, v in zip(*np.unique(labels, return_counts=True))
        },
        "silhouette_score": cluster_comparison[
            str(selected_cluster_count)
        ]["silhouette_score"],
        "profiles": cluster_profiles,
    }

    cluster_data = cluster_frame.copy()
    cluster_data["cluster"] = labels
    cluster_data.to_csv(
        ARTIFACTS / "farm_cluster_assignments.csv", index=False
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["field_size_ha"], df["rainfall_mm"], c=labels)
    ax.set_xlabel("Field size (hectares)")
    ax.set_ylabel("Rainfall (mm)")
    ax.set_title("Farm clusters")
    fig.tight_layout()
    fig.savefig(PLOTS / "farm_clusters.png")
    plt.close(fig)

    (ARTIFACTS / "metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    for task in ("regression", "classification", "clustering"):
        (ARTIFACTS / f"{task}_metrics.json").write_text(
            json.dumps(results[task], indent=2), encoding="utf-8"
        )

    print("Musanze Harvest Intelligence Service")
    print("Verification code: AI-A2-6657")
    print(f"Rows after duplicate removal: {len(df)}")
    print(f"Best regression model: {best_regressor_name}")
    print(f"Best classification model: {best_classifier_name}")
    print("Results saved to artifacts/metrics.json")

    return results


if __name__ == "__main__":
    train_and_evaluate()


