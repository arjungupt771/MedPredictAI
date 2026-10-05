"""
MedPredict AI
UCI Heart Disease Benchmark Configuration
"""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "dataset" / "uci_heart_disease" / "raw"
RESULTS_DIR = BASE_DIR / "experiments" / "uci_heart_disease"

RANDOM_SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5


DATASETS = {
    "cleveland": DATA_DIR / "processed.cleveland.data",
    "hungarian": DATA_DIR / "processed.hungarian.data",
    "switzerland": DATA_DIR / "processed.switzerland.data",
    "va": DATA_DIR / "processed.va.data",
}


FEATURE_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

TARGET_NAME = "target"


# Categorical variables from the UCI documentation.
CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "thal",
]


NUMERIC_FEATURES = [
    feature
    for feature in FEATURE_NAMES
    if feature not in CATEGORICAL_FEATURES
]