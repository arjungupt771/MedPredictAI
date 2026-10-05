"""
Visualization utilities for UCI Heart Disease benchmark.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    RocCurveDisplay,
    PrecisionRecallDisplay,
)


def save_evaluation_plots(
    model,
    X_test,
    y_test,
    output_dir: Path,
    model_name: str,
):
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ROC curve
    RocCurveDisplay.from_estimator(
        model,
        X_test,
        y_test,
    )

    plt.title(
        f"ROC Curve — {model_name}"
    )

    plt.tight_layout()

    plt.savefig(
        output_dir / "roc_curve.png",
        dpi=200,
    )

    plt.close()

    # Precision-recall curve
    PrecisionRecallDisplay.from_estimator(
        model,
        X_test,
        y_test,
    )

    plt.title(
        f"Precision-Recall Curve — {model_name}"
    )

    plt.tight_layout()

    plt.savefig(
        output_dir / "precision_recall_curve.png",
        dpi=200,
    )

    plt.close()

    # Calibration
    CalibrationDisplay.from_estimator(
        model,
        X_test,
        y_test,
        n_bins=10,
    )

    plt.title(
        f"Calibration Curve — {model_name}"
    )



    plt.tight_layout()

    plt.savefig(
        output_dir / "calibration_curve.png",
        dpi=200,
    )

    plt.close()