# Mugabo Alvin Marvin

- Registration number: 24/26657
- Institutional username: ug2426657@ines.ac.rw
- Verification code: AI-A2-6657

## Musanze Harvest Intelligence Service

**AI Assignment 2**

- Repository URL: https://github.com/Alvinmarv/AI_A2_2426657
- Final commit hash: **[ADD AFTER FINAL COMMIT]**
- Tested environment: Python 3.14.7, Windows (exact Windows build not recorded)

## 1. Project Overview

Musanze Harvest Intelligence Service is a machine-learning application designed to analyze farm data and provide three types of predictions:

1. **Crop-yield prediction:** Estimates crop yield in tons.
2. **High-risk classification:** Estimates whether a farm is in the high-risk category.
3. **Farm clustering:** Groups farms using the K-Means clustering algorithm.

The application provides a REST API built with FastAPI and a command-line prediction script.

## 2. Technologies Used

- Python
- Pandas and NumPy for data handling
- Scikit-learn for machine learning
- Matplotlib for visualizations
- Joblib for saving and loading trained models
- FastAPI and Uvicorn for the REST API
- Pytest for automated testing

## 3. Dataset

The dataset is synthetic and generated deterministically for this assignment.
It is not a source of real farm observations and should not be represented as
such. The dataset is located at `data/AI_A2_2426657.csv`.

Features include:
- Rainfall in millimetres
- Average temperature in degrees Celsius
- Soil moisture percentage
- Fertilizer quantity in kilograms
- Field size in hectares
- Season
- Cooperative support

The target variables are crop yield (`yield_tons`) and high-risk status (`high_risk`).

The preprocessing pipeline removes exact duplicate rows and handles missing feature values during model training and prediction.

## 4. Installation

The verified environment is Python 3.14.7 on Windows. Open Git Bash in the
extracted `AI_A2_2426657` project directory.

Create and activate a virtual environment, then install the dependencies:

On Windows Git Bash:

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Reproduce the Dataset and Models

Regenerate the personalized CSV and fingerprint, train models, run tests, and
write an artifact manifest:

```bash
python run_all.py --student-id "24/26657" --seed 6657
```

The script rejects a student ID or seed that does not match this project. To
run only the tests:

```bash
python -m pytest -q
```

## 6. Run the API

Start the FastAPI server:

```bash
python -m uvicorn app:app --reload
```

Open these URLs in a browser:

- API root: http://127.0.0.1:8000/
- Health check: http://127.0.0.1:8000/health
- Interactive API documentation: http://127.0.0.1:8000/docs

### Available endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Service information |
| GET | `/health` | Check service and model availability |
| GET | `/model-info` | Selected algorithms, model artifacts, and features |
| POST | `/predict/yield` | Predict crop yield |
| POST | `/predict/risk` | Predict high-risk status |
| POST | `/predict/cluster` | Assign a farm cluster |

### Example request

Send the following JSON to any of the three prediction endpoints:

```json
{
  "rainfall_mm": 870.6,
  "avg_temp_c": 20.0,
  "soil_moisture_pct": 40.0,
  "fertilizer_kg": 120.0,
  "field_size_ha": 1.5,
  "season": "A",
  "cooperative_support": 1
}
```

The request must satisfy the validation rules defined by the API. Invalid
request bodies return HTTP 422.

Example calls from Git Bash:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/model-info
curl -X POST http://127.0.0.1:8000/predict/yield \
  -H "Content-Type: application/json" --data @samples/valid.json
```

## 7. Command-Line Predictions

After training the models, run:

```bash
python predict.py
```

To use a JSON input file:

```bash
python predict.py samples/valid.json
```

The script prints the estimated yield, risk classification, risk probability, and cluster as JSON.

## 8. Automated Tests

Run the automated tests:

```bash
python -m pytest -q
```

The tests cover health and model-info endpoints, all three prediction routes,
invalid requests, missing models, deterministic data generation, and dataset
range validation.

## 9. Evaluation Results

The current evaluation run produced the following results:

### Regression

The mean baseline and selected Linear Regression model achieved:

- Mean baseline: MAE 1.932, RMSE 2.450, R² -0.001
- Linear Regression: MAE 0.891 tons, RMSE 1.173 tons, R² 0.771
- Random Forest: MAE 1.279 tons, RMSE 1.546 tons, R² 0.601
- Selection: lowest test RMSE. Residuals (actual minus predicted) have mean
  -0.124 tons, indicating slight average overprediction on this holdout.

### Classification

The majority baseline, Logistic Regression, and selected Random Forest achieved:

- Majority baseline: accuracy 0.510, F1 0.000, ROC-AUC 0.500
- Logistic Regression: precision 0.828, recall 0.960, F1 0.889, ROC-AUC 0.983
- Random Forest: precision 0.893, recall 1.000, F1 0.943, ROC-AUC 0.996
- Random Forest confusion matrix (true rows, predicted columns):
  `[[23, 3], [0, 25]]`; selection is based on highest test F1.

### Clustering

K-Means compares K=2 through K=6 using silhouette score. K=2 is selected with
score 0.160; scores for K=3, 4, 5, and 6 are 0.142, 0.144, 0.137, and 0.142.
The low selected score indicates that the groups overlap substantially;
clusters are descriptive groupings, not validated farm categories.

Metrics may change if the dataset, preprocessing, random seed, or model configuration changes.

## 10. Project Structure

```text
AI_A2_2426657/
├── app.py
├── predict.py
├── run_all.py
├── requirements.txt
├── README.md
├── docs/
│   └── AI_A2_2426657_REPORT.md
├── evidence/
│   └── AI_USE.md
├── data/
│   ├── AI_A2_2426657.csv
│   └── generate_data.py
├── src/
│   ├── __init__.py
│   └── pipeline.py
├── models/
│   ├── farm_clusters.joblib
│   ├── high_risk_classifier.joblib
│   └── yield_regression.joblib
├── artifacts/
│   ├── data_fingerprint.txt
│   ├── farm_cluster_assignments.csv
│   ├── metrics.json
│   ├── regression_metrics.json
│   ├── classification_metrics.json
│   ├── clustering_metrics.json
│   ├── run_manifest.json
│   └── plots/
└── tests/
  └── test_api.py
```

The Markdown report and AI-use disclosure are source drafts. The required
final report PDF, continuous demo MP4, test-log PDF, signed declaration PDF,
and `AI_A2_2426657.zip` have not been produced yet.

## 11. Limitations and Responsible Use

All results use a small synthetic dataset and a single randomized holdout
split. They do not establish real-world accuracy, causal relationships, fairness
across farmer groups, or suitability for operational decisions. The
classification target is derived from yield per hectare, and the low clustering
silhouette score indicates weakly separated groups. Treat outputs as
experimental estimates only; a qualified human should check local conditions
and independent field evidence before decisions affecting farmers.

## 12. Verification

- Project: Musanze Harvest Intelligence Service
- Registration number: 24/26657
- Verification code: AI-A2-6657
- Dataset rows after cleaning: 255
- Dataset SHA-256: `ba50b841d0210e5d158289c4ff206d953e96a446ec325d714f5c88ead892e6c2`
- Latest generated outputs: see `artifacts/run_manifest.json`
