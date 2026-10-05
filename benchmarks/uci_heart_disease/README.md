# UCI Heart Disease Benchmark

This directory contains an independent clinical tabular benchmark for MedPredict AI using the UCI Heart Disease dataset.

## Purpose

The benchmark evaluates conventional machine-learning models on a real-world clinical dataset independently from the primary MedPredict AI symptom-classification system.

The benchmark does **not** replace the existing 20-disease MedPredict AI model.

## Dataset

The UCI Heart Disease dataset contains four databases:

* Cleveland
* Hungarian Institute of Cardiology
* Switzerland
* VA Long Beach

The commonly used processed representation contains 13 predictive features and a target representing the presence/severity of heart disease.

Source:

UCI Machine Learning Repository — Heart Disease

https://archive.ics.uci.edu/dataset/45/heart+disease

DOI:

10.24432/C52P4X

## Target transformation

The original target contains values:

* `0` — absence of heart disease
* `1–4` — presence/severity of heart disease

For this benchmark, the task is converted to binary classification:

```text
0 → No heart disease
1–4 → Heart disease present
```

This follows the binary classification formulation described in the original UCI documentation.

## Features

The 13 predictive features are:

* age
* sex
* cp
* trestbps
* chol
* fbs
* restecg
* thalach
* exang
* oldpeak
* slope
* ca
* thal

## Models

The benchmark compares:

1. Logistic Regression
2. Random Forest
3. XGBoost

The selected model is subsequently evaluated with probability calibration.

## Metrics

The benchmark reports:

* Accuracy
* Precision
* Recall
* F1
* ROC-AUC
* Brier score
* Confusion matrix

Five-fold stratified cross-validation is also performed.

## Missing values

The original datasets contain missing values represented by `?`.

Missing values are handled inside the preprocessing pipeline using training-data-derived imputation.

No imputation is performed before the train/test split.

This prevents preprocessing leakage from the test set into training.

## Running the benchmark

From the project root:

```bash
python -m benchmarks.uci_heart_disease.train --dataset cleveland
```

Other datasets:

```bash
python -m benchmarks.uci_heart_disease.train --dataset hungarian

python -m benchmarks.uci_heart_disease.train --dataset switzerland

python -m benchmarks.uci_heart_disease.train --dataset va
```

Results are written to:

```text
experiments/uci_heart_disease/
```

## Important limitation

This benchmark is a machine-learning evaluation exercise and is **not a clinical diagnostic system**.

The dataset is relatively small and comes from historical clinical cohorts. Performance measured on this dataset should not be interpreted as evidence of clinical effectiveness or suitability for real-world medical diagnosis.

The benchmark is intended to evaluate model-development techniques such as preprocessing, model comparison, calibration and evaluation methodology.

The Switzerland cohort is especially small and highly imbalanced, with only eight negative samples. Accuracy and F1 should therefore be interpreted cautiously, and ROC-AUC can vary substantially between cross-validation folds. A single test split should not be presented as a general estimate of performance.

## Attribution

Please retain the original UCI documentation and attribution information when redistributing or publishing results based on this dataset.

The UCI Heart Disease dataset is maintained by the UCI Machine Learning Repository.
