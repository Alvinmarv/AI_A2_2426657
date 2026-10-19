# Musanze Harvest Intelligence Service: Technical Report

- **Assignment:** AI Assignment 2
- **Student:** Mugabo Alvin Marvin
- **Registration:** 24/26657
- **Institutional username:** ug2426657@ines.ac.rw
- **Verification code:** AI-A2-6657
- **Seed:** 6657
- **Repository:** https://github.com/Alvinmarv/AI_A2_2426657
- **Source commit:** `e8af733a4e25f384464a0ff289f4421823b1dff4`
- **Repository HEAD when this report was prepared:** `a2f8eea2cacb54a7f878167083e4ddb431634ce8`
- **Dataset SHA-256:** `ba50b841d0210e5d158289c4ff206d953e96a446ec325d714f5c88ead892e6c2`

## Problem and data

This project predicts synthetic crop yield, classifies a proxy high-risk label, and groups records by input characteristics. The CSV is generated for this assignment; it contains no real farm observations.

The generator uses NumPy's `default_rng(6657)` and 255 base rows. Yield in tons is calculated as:

`0.004*rainfall + 0.06*moisture + 0.02*fertilizer + 1.30*field_size - 0.25*abs(temperature - 18) + 0.70*support + season_effect + noise`

where season effect is `+0.50` for season A and `-0.20` for season B, noise is normally distributed with mean 0 and standard deviation 0.80, and yield is clipped to a minimum of 0.50. High risk is assigned when yield per hectare is below the median among base records. The generator then injects 13 missing values in each of soil moisture and fertilizer and appends three exact duplicate rows, producing 258 CSV rows. Cleaning removes those three duplicate records, leaving 255 rows. The stable CSV fingerprint is recorded in `artifacts/data_fingerprint.txt`.

Schema and range validation runs before training. It checks required columns, allowed missingness, numeric types, feature ranges, season values, and binary labels. Only soil moisture and fertilizer may be missing.

## Methods

For supervised tasks, a fixed 80/20 holdout split uses seed 6657; classification is stratified. Imputation, one-hot encoding, and scaling are held inside scikit-learn pipelines and fit on training data only. Numeric missing values use the median; categorical values use the most frequent value; unknown categories are ignored. Yield is excluded from the high-risk features because the risk label is derived from yield per hectare.

Regression compares a mean DummyRegressor baseline, Linear Regression, and Random Forest. Lowest test RMSE selects the saved model. Classification compares a majority DummyClassifier baseline, Logistic Regression, and class-balanced Random Forest. The primary selection criterion is test F1. It also records precision, recall, accuracy, ROC-AUC, confusion matrices, and false-positive/false-negative counts.

Clustering uses only the seven input features, without either target. A train/test claim is not made for this exploratory unsupervised analysis: the preprocessor and K-Means models fit all 255 cleaned rows. K values 2 through 6 are compared by silhouette score, and the highest-scoring K is selected. Feature means and season shares describe each cluster.

## Results

### Regression

| Model | MAE (tons) | RMSE (tons) | R-squared |
|---|---:|---:|---:|
| Mean baseline | 1.932 | 2.450 | -0.001 |
| Linear Regression | 0.891 | 1.173 | 0.771 |
| Random Forest | 1.279 | 1.546 | 0.601 |

Linear Regression has the lowest RMSE and is selected. Its holdout residual mean (actual minus predicted) is -0.124 tons, suggesting slight average overprediction; residual standard deviation is 1.166 tons and maximum absolute error is 4.349 tons. These are holdout estimates on synthetic data.

### Classification

| Model | Precision | Recall | F1 | ROC-AUC | Accuracy |
|---|---:|---:|---:|---:|---:|
| Majority baseline | 0.000 | 0.000 | 0.000 | 0.500 | 0.510 |
| Logistic Regression | 0.828 | 0.960 | 0.889 | 0.983 | 0.882 |
| Random Forest | 0.893 | 1.000 | 0.943 | 0.996 | 0.941 |

Random Forest is selected by F1. Its confusion matrix, with true labels as rows and predictions as columns, is `[[23, 3], [0, 25]]`: three lower-risk records are false positives and no high-risk records are false negatives in this holdout. This result does not establish real-world performance, especially because the label is mechanically derived from synthetic yield.

### Clustering

| K | Silhouette | Inertia |
|---:|---:|---:|
| 2 | 0.160 | 1394.662 |
| 3 | 0.142 | 1270.609 |
| 4 | 0.144 | 1161.433 |
| 5 | 0.137 | 1088.186 |
| 6 | 0.142 | 1019.446 |

K=2 has the highest silhouette and is selected. Cluster 0 has 127 records and mean cooperative support 0.00; cluster 1 has 128 records and mean support 1.00. Other feature means are similar: rainfall is about 973 versus 1012 mm, temperature 18.51 versus 19.43 C, soil moisture 51.35 versus 49.62%, fertilizer 119.85 versus 121.75 kg, and field size 1.98 versus 2.00 ha. The best silhouette is low (0.160), so these clusters overlap and should not be presented as robust natural farm segments.

## Artifacts and reproduction

Run from the extracted project folder:

```bash
python run_all.py --student-id "24/26657" --seed 6657
```

The command regenerates the data and fingerprint, trains and evaluates models, runs pytest, and records generated files and the fingerprint in `artifacts/run_manifest.json`. Metrics are in `artifacts/metrics.json`; plots and cluster assignments are under `artifacts/`; trained estimators are in `models/`. The API and CLI are documented in the root README.

## Limitations and responsible use

The dataset is small and synthetic, and metrics come from one fixed split. They do not demonstrate generalization, causal effects, fairness, or value for real farms. The risk label is a constructed proxy, the residuals can be substantial for individual observations, and K-Means separation is weak. Do not use outputs to allocate credit, insurance, resources, or other consequential support. Any future field use requires representative consented data, external validation, bias and privacy assessment, and qualified local human review. Model outputs are decision support only, not agricultural advice.

## Submission status

This report is provided as `docs/AI_A2_2426657_REPORT.pdf`, with its Markdown source and reproducible builder alongside it. The independently produced demo video, test-log PDF, and signed declaration remain separate submission items and must reflect the student's own work.
