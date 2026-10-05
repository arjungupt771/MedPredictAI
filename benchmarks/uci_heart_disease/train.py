
import joblib
import argparse
import json

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
)

from benchmarks.uci_heart_disease.plots import save_evaluation_plots

from .config import (
    DATASETS,
    FEATURE_NAMES,
    RANDOM_SEED,
    TEST_SIZE,
    CV_FOLDS,
    RESULTS_DIR,
)

from .data import (
    load_dataset,
    dataset_summary,
)

from .models import build_models

from .evaluate import (
    evaluate_binary_classifier,
    save_results,
)


def run_benchmark(dataset_name: str):

    if dataset_name not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Choose from: {list(DATASETS)}"
        )

    dataset_path = DATASETS[dataset_name]

    print("=" * 70)
    print("MedPredict AI — UCI Heart Disease Benchmark")
    print("=" * 70)

    print(f"\nDataset: {dataset_name}")
    print(f"File: {dataset_path}")

    df = load_dataset(dataset_path)

    summary = dataset_summary(df)

    print("\nDataset summary:")
    print(json.dumps(summary, indent=2))

    X = df[FEATURE_NAMES]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    print(
        f"\nTrain samples: {len(X_train)}"
        f"\nTest samples:  {len(X_test)}"
    )

    models = build_models(X_train.columns)

    results = {
        "dataset": dataset_name,
        **summary,
        "dataset_summary": summary,
        "removed_all_missing_features": [],
    }

    # ------------------------------------------------------------------
    # Cross-validation model selection
    # ------------------------------------------------------------------

    print("\nRunning 5-fold cross-validation for model selection...")

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    cv_results = {}

    for name, model in models.items():

        print(f"\nCross-validating {name}...")

        cv_scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=[
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
            ],
            n_jobs=-1,
        )

        cv_results[name] = {
            "accuracy_mean": float(
                cv_scores["test_accuracy"].mean()
            ),
            "accuracy_std": float(
                cv_scores["test_accuracy"].std()
            ),
            "precision_mean": float(
                cv_scores["test_precision"].mean()
            ),
            "precision_std": float(
                cv_scores["test_precision"].std()
            ),
            "recall_mean": float(
                cv_scores["test_recall"].mean()
            ),
            "recall_std": float(
                cv_scores["test_recall"].std()
            ),
            "f1_mean": float(
                cv_scores["test_f1"].mean()
            ),
            "f1_std": float(
                cv_scores["test_f1"].std()
            ),
            "roc_auc_mean": float(
                cv_scores["test_roc_auc"].mean()
            ),
            "roc_auc_std": float(
                cv_scores["test_roc_auc"].std()
            ),
        }

        print(
            f"ROC-AUC: "
            f"{cv_results[name]['roc_auc_mean']:.4f} "
            f"+/- "
            f"{cv_results[name]['roc_auc_std']:.4f}"
        )

    results["cross_validation"] = cv_results

    # Select model only from training-data cross-validation.
    best_name = max(
        cv_results,
        key=lambda name: cv_results[name]["roc_auc_mean"],
    )
    results["selected_model"] = best_name

    print(
        f"\nSelected model using cross-validation ROC-AUC: "
        f"{best_name}"
    )

    best_model = models[best_name]

    print(f"\nTraining selected model: {best_name}")
    best_model.fit(
        X_train,
        y_train,
    )

    final_metrics = evaluate_binary_classifier(
        best_model,
        X_test,
        y_test,
    )
    results["final_test"] = final_metrics
    # ------------------------------------------------------------------
    # Probability calibration
    # ------------------------------------------------------------------

    print("\nCalibrating selected model...")

    calibrated_model = CalibratedClassifierCV(
        estimator=models[best_name],
        method="sigmoid",
        cv=5,
    )

    calibrated_model.fit(
        X_train,
        y_train,
    )

    model_output_dir = RESULTS_DIR / dataset_name
    model_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        best_model,
        model_output_dir / "best_model.joblib",
        )

    joblib.dump(
        calibrated_model,
        model_output_dir / "calibrated_model.joblib",
        )

    print(
        f"\nModels saved to:\n"
        f"{model_output_dir}"
    )

    calibrated_metrics = evaluate_binary_classifier(
        calibrated_model,
        X_test,
        y_test,
    )
    save_evaluation_plots(
        calibrated_model,
        X_test,
        y_test,
        RESULTS_DIR / dataset_name,
        "Calibrated selected model",
    )

    results["calibrated_model"] = calibrated_metrics

    print("\nCalibrated model:")
    print(
        f"Accuracy : {calibrated_metrics['accuracy']:.4f}\n"
        f"Precision: {calibrated_metrics['precision']:.4f}\n"
        f"Recall   : {calibrated_metrics['recall']:.4f}\n"
        f"F1       : {calibrated_metrics['f1']:.4f}\n"
        f"ROC-AUC  : {calibrated_metrics['roc_auc']:.4f}\n"
        f"Brier    : {calibrated_metrics['brier_score']:.4f}"
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    output_path = (
        RESULTS_DIR /
        f"{dataset_name}_results.json"
    )

    save_results(
        results,
        output_path,
    )

    print(
        f"\nResults saved to:\n"
        f"{output_path}"
    )

    return results


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default="cleveland",
        choices=list(DATASETS.keys()),
        help="UCI dataset to benchmark",
    )

    args = parser.parse_args()

    run_benchmark(
        args.dataset
    )


if __name__ == "__main__":
    main()