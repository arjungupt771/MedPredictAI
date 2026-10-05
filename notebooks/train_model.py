"""
MedPredict AI - Model Training Pipeline (v2.1)
Trains RF + XGBoost, evaluates with 5-fold CV, calibrates probabilities,
saves per-class metrics, confusion matrix, and full model metadata.
"""
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
from experiment_logger import log_experiment

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

RANDOM_SEED = 42
DATASET_PATH = Path("./dataset/diseases.csv")
MODEL_DIR = Path("./models")
MODEL_DIR.mkdir(exist_ok=True)

# ── Load Data ────────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(DATASET_PATH)
dataset_hash = hashlib.md5(DATASET_PATH.read_bytes()).hexdigest()[:12]

symptom_cols = [c for c in df.columns if c != "disease"]
X = df[symptom_cols].values
y = df["disease"].values

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=RANDOM_SEED, stratify=y_encoded
)
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ── Optional external / held-out evaluation set ───────────────────────────────
# Drop a real-world or public symptom-disease CSV at dataset/external_test.csv
# with the same symptom columns + a 'disease' column, and this block will
# evaluate on it automatically instead of only on the synthetic split.
EXTERNAL_PATH = Path("./dataset/external_test.csv")
external_df = None
if EXTERNAL_PATH.exists():
    external_df = pd.read_csv(EXTERNAL_PATH)
    missing_cols = set(symptom_cols) - set(external_df.columns)
    if missing_cols:
        print(f"⚠️  External test set missing columns {missing_cols}, skipping.")
        external_df = None
    else:
        print(f"Found external eval set: {external_df.shape[0]} rows")


def evaluate(model, X_te, y_te, average_extra=True):
    pred = model.predict(X_te)
    result = {
        "accuracy": round(accuracy_score(y_te, pred) * 100, 2),
        "precision_weighted": round(precision_score(y_te, pred, average="weighted", zero_division=0) * 100, 2),
        "recall_weighted": round(recall_score(y_te, pred, average="weighted", zero_division=0) * 100, 2),
        "f1_weighted": round(f1_score(y_te, pred, average="weighted", zero_division=0) * 100, 2),
        "f1_macro": round(f1_score(y_te, pred, average="macro", zero_division=0) * 100, 2),
    }
    if average_extra:
        report = classification_report(
            y_te, pred, target_names=le.classes_, output_dict=True, zero_division=0
        )
        result["per_class"] = {
            cls: {k: round(v, 4) for k, v in report[cls].items() if k != "support"}
            for cls in le.classes_
        }
        cm = confusion_matrix(y_te, pred, labels=range(len(le.classes_)))
        result["confusion_matrix"] = cm.tolist()
        result["confusion_matrix_labels"] = list(le.classes_)
    return result


# ── Train Models ───────────────────────────────────────────────────────────────
results = {}
fitted = {}

print("\nTraining Random Forest...")
rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=RANDOM_SEED, n_jobs=-1)
rf.fit(X_train, y_train)
fitted["RandomForest"] = rf
results["RandomForest"] = evaluate(rf, X_test, y_test)
log_experiment(
    "rf_baseline",
    hyperparameters={"n_estimators": 200, "max_depth": 15, "random_state": RANDOM_SEED},
    metrics={k: v for k, v in results["RandomForest"].items() if k in
             ("accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "f1_macro")},
    notes="Uncalibrated RandomForest on synthetic dataset.",
)
print(f"  Accuracy: {results['RandomForest']['accuracy']}% | Macro-F1: {results['RandomForest']['f1_macro']}%")

print("\nTraining XGBoost...")
xgb = XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    eval_metric="mlogloss", random_state=RANDOM_SEED, n_jobs=-1,
)
xgb.fit(X_train, y_train)
fitted["XGBoost"] = xgb
results["XGBoost"] = evaluate(xgb, X_test, y_test)
log_experiment(
    "xgb_baseline",
    hyperparameters={"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1, "random_state": RANDOM_SEED},
    metrics={k: v for k, v in results["XGBoost"].items() if k in
             ("accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "f1_macro")},
    notes="Uncalibrated XGBoost on synthetic dataset.",
)
print(f"  Accuracy: {results['XGBoost']['accuracy']}% | Macro-F1: {results['XGBoost']['f1_macro']}%")

# ── Pick Best Model (by weighted F1, same rule as before) ────────────────────
best_name = max(results, key=lambda k: results[k]["f1_weighted"])
best_model_uncalibrated = fitted[best_name]
print(f"\nBest model: {best_name}")

# ── 5-Fold Stratified Cross-Validation on the winning architecture ───────────
print("\nRunning 5-fold stratified CV on the winning model type...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
cv_estimator = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=RANDOM_SEED, n_jobs=-1) \
    if best_name == "RandomForest" else \
    XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, eval_metric="mlogloss",
                  random_state=RANDOM_SEED, n_jobs=-1)

cv_scores = cross_validate(
    cv_estimator, X, y_encoded, cv=skf,
    scoring={"accuracy": "accuracy", "f1_weighted": "f1_weighted", "f1_macro": "f1_macro"},
)
cv_summary = {
    "accuracy_mean": round(cv_scores["test_accuracy"].mean() * 100, 2),
    "accuracy_std": round(cv_scores["test_accuracy"].std() * 100, 2),
    "f1_weighted_mean": round(cv_scores["test_f1_weighted"].mean() * 100, 2),
    "f1_weighted_std": round(cv_scores["test_f1_weighted"].std() * 100, 2),
    "f1_macro_mean": round(cv_scores["test_f1_macro"].mean() * 100, 2),
    "f1_macro_std": round(cv_scores["test_f1_macro"].std() * 100, 2),
}
print(f"  CV Accuracy: {cv_summary['accuracy_mean']}% ± {cv_summary['accuracy_std']}%")

# ── Probability Calibration ────────────────────────────────────────────────────
# NOTE: SHAP TreeExplainer needs the raw tree ensemble, so we keep
# `best_model_uncalibrated` for explainability and ship a *separate*
# calibrated wrapper purely for confidence scores.
print("\nCalibrating probabilities (sigmoid, 5-fold)...")
calibrated = CalibratedClassifierCV(
    estimator=type(best_model_uncalibrated)(**best_model_uncalibrated.get_params()),
    method="sigmoid",
    cv=5,
)
calibrated.fit(X_train, y_train)
calibrated_eval = evaluate(calibrated, X_test, y_test, average_extra=False)
log_experiment(
    "calibrated_" + best_name.lower(),
    hyperparameters={"base_model": best_name, "calibration_method": "sigmoid", "cv": 5},
    metrics={
        "accuracy": calibrated_eval["accuracy"],
        "cv_accuracy_mean": cv_summary["accuracy_mean"],
        "cv_accuracy_std": cv_summary["accuracy_std"],
        "f1_weighted": results[best_name]["f1_weighted"],
        "f1_macro": results[best_name]["f1_macro"],
    },
    notes="Sigmoid-calibrated wrapper around the winning model; this is what predictor.py loads for confidence scores.",
)

# External evaluation, if provided
external_eval = None
if external_df is not None:
    X_ext = external_df[symptom_cols].values
    y_ext = le.transform(external_df["disease"].values)
    external_eval = evaluate(calibrated, X_ext, y_ext)
    print(f"  External held-out accuracy: {external_eval['accuracy']}%")

# ── Save Artifacts ─────────────────────────────────────────────────────────────
joblib.dump(best_model_uncalibrated, MODEL_DIR / "model.pkl")          # for SHAP
joblib.dump(calibrated, MODEL_DIR / "calibrated_model.pkl")            # for confidence
joblib.dump(rf, MODEL_DIR / "rf_model.pkl")
joblib.dump(xgb, MODEL_DIR / "xgb_model.pkl")
joblib.dump(le, MODEL_DIR / "label_encoder.pkl")
joblib.dump(symptom_cols, MODEL_DIR / "symptoms.pkl")

stats = {name: data for name, data in results.items()}
stats["best_model"] = best_name
stats["cross_validation"] = cv_summary
stats["calibrated_test_accuracy"] = calibrated_eval["accuracy"]
if external_eval:
    stats["external_evaluation"] = external_eval
with open(MODEL_DIR / "model_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

metadata = {
    "model_version": "2.1.0",
    "model_type": best_name,
    "calibration_method": "sigmoid",
    "dataset_version": "1.0.0",
    "dataset_hash": dataset_hash,
    "dataset_is_synthetic": external_eval is None,
    "training_date": datetime.now(timezone.utc).isoformat(),
    "python_version": sys.version.split()[0],
    "platform": platform.platform(),
    "sklearn_version": sklearn.__version__,
    "xgboost_version": xgboost.__version__,
    "features": len(symptom_cols),
    "classes": len(le.classes_),
    "random_seed": RANDOM_SEED,
}
with open(MODEL_DIR / "model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\nAll artifacts saved to models/")
print(f"Symptoms: {len(symptom_cols)} | Diseases: {len(le.classes_)}")
print("\nHonest summary line for README:")
print(
    f"Achieved {results[best_name]['accuracy']}% held-out accuracy and "
    f"{results[best_name]['f1_weighted']}% weighted F1 (macro-F1 "
    f"{results[best_name]['f1_macro']}%) on a synthetic {len(le.classes_)}-class "
    f"symptom dataset, with 5-fold cross-validation achieving "
    f"{cv_summary['accuracy_mean']}% ± {cv_summary['accuracy_std']}% accuracy."
)