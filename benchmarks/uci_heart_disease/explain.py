"""
SHAP explainability for the UCI Heart Disease benchmark.

Usage:
    python -m benchmarks.uci_heart_disease.explain --dataset cleveland
    python -m benchmarks.uci_heart_disease.explain --dataset hungarian
    python -m benchmarks.uci_heart_disease.explain --dataset switzerland
    python -m benchmarks.uci_heart_disease.explain --dataset va
    python -m benchmarks.uci_heart_disease.explain --all
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from benchmarks.uci_heart_disease.config import (
    EXPERIMENTS_DIR,
)
from benchmarks.uci_heart_disease.data import load_dataset


warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DATASETS = (
    "cleveland",
    "hungarian",
    "switzerland",
    "va",
)

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


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

def get_model_path(dataset: str) -> Path:
    """
    Return the calibrated model path.
    """

    return (
        EXPERIMENTS_DIR
        / dataset
        / "calibrated_model.joblib"
    )


def get_output_dir(dataset: str) -> Path:
    """
    Directory for SHAP outputs.
    """

    output_dir = (
        EXPERIMENTS_DIR
        / "explainability"
        / dataset
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_dir


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------

def load_model(dataset: str):
    """
    Load the already-trained calibrated model.
    """

    model_path = get_model_path(dataset)

    if not model_path.exists():
        raise FileNotFoundError(
            f"\nCalibrated model not found:\n"
            f"{model_path}\n\n"
            f"Train the {dataset} benchmark first."
        )

    print(
        f"Loading model:\n{model_path}"
    )

    return joblib.load(model_path)


# ---------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------

def load_features(dataset: str) -> pd.DataFrame:
    """
    Load the UCI dataset and return the 13 clinical features.
    """

    X, _ = load_dataset(dataset)

    missing_features = [
        feature
        for feature in FEATURE_NAMES
        if feature not in X.columns
    ]

    if missing_features:
        raise ValueError(
            "Dataset is missing expected features: "
            + ", ".join(missing_features)
        )

    return X[FEATURE_NAMES].copy()


# ---------------------------------------------------------------------
# Prediction function
# ---------------------------------------------------------------------

def build_prediction_function(model):
    """
    Build a prediction function compatible with SHAP.

    The complete saved model is used, including:
        preprocessing
        model
        calibration
    """

    def predict(data):

        if isinstance(data, pd.DataFrame):
            frame = data.copy()
        else:
            frame = pd.DataFrame(
                data,
                columns=FEATURE_NAMES,
            )

        probabilities = model.predict_proba(
            frame
        )

        return probabilities[:, 1]

    return predict


# ---------------------------------------------------------------------
# Background data
# ---------------------------------------------------------------------

def create_background_data(
    X: pd.DataFrame,
    max_background: int = 50,
) -> pd.DataFrame:
    """
    Create a small background dataset for SHAP.

    A smaller background set keeps permutation SHAP
    computationally manageable.
    """

    if len(X) <= max_background:
        return X.copy()

    return X.sample(
        n=max_background,
        random_state=42,
    )


# ---------------------------------------------------------------------
# SHAP calculation
# ---------------------------------------------------------------------

def calculate_shap_values(
    model,
    X: pd.DataFrame,
):
    """
    Calculate model-agnostic SHAP values.

    We intentionally use permutation SHAP because the
    saved benchmark models include preprocessing and
    calibration.
    """

    prediction_function = build_prediction_function(
        model
    )

    background = create_background_data(X)

    print(
        f"SHAP background samples: "
        f"{len(background)}"
    )

    print(
        "Creating SHAP permutation explainer..."
    )

    masker = shap.maskers.Independent(
        background
    )

    explainer = shap.Explainer(
        prediction_function,
        masker,
        algorithm="permutation",
    )

    print(
        f"Calculating SHAP values for "
        f"{len(X)} samples..."
    )

    max_evals = max(
        100,
        2 * len(FEATURE_NAMES) + 1,
    )

    shap_values = explainer(
        X,
        max_evals=max_evals,
    )

    return shap_values


# ---------------------------------------------------------------------
# Global feature importance
# ---------------------------------------------------------------------

def calculate_global_importance(
    shap_values,
    X: pd.DataFrame,
):
    """
    Calculate mean absolute SHAP importance.
    """

    values = np.asarray(
        shap_values.values
    )

    mean_absolute = np.mean(
        np.abs(values),
        axis=0,
    )

    importance = pd.DataFrame(
        {
            "feature": X.columns,
            "mean_abs_shap": mean_absolute,
        }
    )

    return importance.sort_values(
        "mean_abs_shap",
        ascending=False,
    )


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------

def save_summary_plot(
    shap_values,
    output_dir: Path,
):
    """
    Save SHAP beeswarm summary plot.
    """

    output_path = (
        output_dir
        / "shap_summary.png"
    )

    print(
        f"Saving: {output_path}"
    )

    plt.figure()

    shap.plots.beeswarm(
        shap_values,
        max_display=len(FEATURE_NAMES),
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def save_bar_plot(
    shap_values,
    output_dir: Path,
):
    """
    Save global SHAP bar plot.
    """

    output_path = (
        output_dir
        / "shap_bar.png"
    )

    print(
        f"Saving: {output_path}"
    )

    plt.figure()

    shap.plots.bar(
        shap_values,
        max_display=len(FEATURE_NAMES),
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# ---------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------

def save_explanations(
    dataset: str,
    model,
    X: pd.DataFrame,
    shap_values,
    output_dir: Path,
):
    """
    Save global and per-sample explanations.
    """

    importance = calculate_global_importance(
        shap_values,
        X,
    )

    values = np.asarray(
        shap_values.values
    )

    base_values = np.asarray(
        shap_values.base_values
    )

    sample_explanations = []

    for i in range(len(X)):

        contributions = []

        for j, feature in enumerate(
            FEATURE_NAMES
        ):

            raw_value = X.iloc[i][feature]

            if pd.isna(raw_value):
                value = None
            else:
                value = float(raw_value)

            contributions.append(
                {
                    "feature": feature,
                    "value": value,
                    "shap_value": float(
                        values[i, j]
                    ),
                }
            )

        contributions.sort(
            key=lambda item: abs(
                item["shap_value"]
            ),
            reverse=True,
        )

        probability = float(
            model.predict_proba(
                X.iloc[[i]]
            )[0, 1]
        )

        sample_explanations.append(
            {
                "sample_index": int(i),
                "predicted_probability": probability,
                "features": contributions,
            }
        )

    result = {
        "dataset": dataset,
        "model_type": type(model).__name__,
        "n_samples": int(len(X)),
        "n_features": int(len(FEATURE_NAMES)),
        "feature_names": FEATURE_NAMES,
        "global_feature_importance": (
            importance.to_dict(
                orient="records"
            )
        ),
        "sample_explanations": sample_explanations,
    }

    output_path = (
        output_dir
        / "explanations.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            indent=2,
        )

    print(
        f"Saved: {output_path}"
    )


# ---------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------

def print_top_features(
    shap_values,
    X: pd.DataFrame,
    top_n: int = 10,
):
    """
    Print globally important features.
    """

    importance = calculate_global_importance(
        shap_values,
        X,
    )

    print()
    print("=" * 70)
    print(
        "GLOBAL SHAP FEATURE IMPORTANCE"
    )
    print("=" * 70)

    for rank, row in enumerate(
        importance.head(top_n).itertuples(
            index=False
        ),
        start=1,
    ):

        print(
            f"{rank:2d}. "
            f"{row.feature:<12} "
            f"{row.mean_abs_shap:.6f}"
        )

    print("=" * 70)


# ---------------------------------------------------------------------
# Dataset workflow
# ---------------------------------------------------------------------

def explain_dataset(dataset: str):

    print()
    print("=" * 70)
    print(
        "MedPredict AI — UCI Heart Disease "
        f"SHAP: {dataset.upper()}"
    )
    print("=" * 70)

    # Load model
    model = load_model(dataset)

    # Load features
    X = load_features(dataset)

    print()
    print(
        f"Samples : {len(X)}"
    )

    print(
        f"Features: {len(X.columns)}"
    )

    print()
    print(
        "Features:"
    )

    print(
        ", ".join(X.columns)
    )

    # Calculate SHAP
    shap_values = calculate_shap_values(
        model,
        X,
    )

    # Output directory
    output_dir = get_output_dir(
        dataset
    )

    # Plots
    save_summary_plot(
        shap_values,
        output_dir,
    )

    save_bar_plot(
        shap_values,
        output_dir,
    )

    # JSON
    save_explanations(
        dataset,
        model,
        X,
        shap_values,
        output_dir,
    )

    # Console summary
    print_top_features(
        shap_values,
        X,
    )

    print()
    print("=" * 70)
    print(
        f"SHAP analysis completed: "
        f"{dataset}"
    )
    print("=" * 70)

    print(
        f"Output directory:\n"
        f"{output_dir}"
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Generate SHAP explanations "
            "for the UCI Heart Disease benchmark."
        )
    )

    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        help="Dataset to explain.",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Explain all datasets.",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    if not args.dataset and not args.all:
        raise SystemExit(
            "Specify --dataset DATASET "
            "or --all."
        )

    if args.all:
        datasets = DATASETS
    else:
        datasets = [args.dataset]

    for dataset in datasets:
        explain_dataset(dataset)


if __name__ == "__main__":
    main()