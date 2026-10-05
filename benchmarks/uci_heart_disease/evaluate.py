"""
Evaluation utilities for UCI Heart Disease benchmark.
"""

import json
from pathlib import Path

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
)


def evaluate_binary_classifier(model, X_test, y_test):
    """Evaluate a binary classifier."""

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(
            accuracy_score(y_test, predictions), 4
        ),
        "precision": round(
            precision_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            4,
        ),
        "recall": round(
            recall_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            4,
        ),
        "f1": round(
            f1_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            4,
        ),
        "roc_auc": round(
            roc_auc_score(
                y_test,
                probabilities,
            ),
            4,
        ),
        "brier_score": round(
            brier_score_loss(
                y_test,
                probabilities,
            ),
            4,
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
        ).tolist(),
    }

    return metrics


def save_results(results: dict, path: Path):
    """Save benchmark results."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(path, "w") as f:
        json.dump(
            results,
            f,
            indent=2,
        )