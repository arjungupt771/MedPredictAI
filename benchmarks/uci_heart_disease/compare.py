"""
MedPredict AI
UCI Heart Disease Benchmark Comparison

Reads all dataset result JSON files and creates:
    experiments/uci_heart_disease/benchmark_summary.json
    experiments/uci_heart_disease/benchmark_summary.csv
"""

import csv
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

RESULTS_DIR = (
    BASE_DIR
    / "experiments"
    / "uci_heart_disease"
)

DATASETS = [
    "cleveland",
    "hungarian",
    "switzerland",
    "va",
]


def load_results():
    results = []

    for dataset in DATASETS:

        path = RESULTS_DIR / f"{dataset}_results.json"

        if not path.exists():
            print(f"Missing: {path}")
            continue

        with open(path, "r") as f:
            data = json.load(f)

        cv = data["cross_validation"]
        test = data["calibrated_model"]

        results.append({
            "dataset": dataset,
            "samples": data["dataset_summary"]["rows"],
            "missing_values": (
                data["dataset_summary"]["missing_values"]
            ),
            "selected_model": data["selected_model"],

            "cv_accuracy_mean": cv[
                data["selected_model"]
            ]["accuracy_mean"],

            "cv_accuracy_std": cv[
                data["selected_model"]
            ]["accuracy_std"],

            "cv_f1_mean": cv[
                data["selected_model"]
            ]["f1_mean"],

            "cv_f1_std": cv[
                data["selected_model"]
            ]["f1_std"],

            "cv_roc_auc_mean": cv[
                data["selected_model"]
            ]["roc_auc_mean"],

            "cv_roc_auc_std": cv[
                data["selected_model"]
            ]["roc_auc_std"],

            "test_accuracy": test["accuracy"],
            "test_precision": test["precision"],
            "test_recall": test["recall"],
            "test_f1": test["f1"],
            "test_roc_auc": test["roc_auc"],
            "test_brier": test["brier_score"],
        })

    return results


def save_json(results):

    output = RESULTS_DIR / "benchmark_summary.json"

    with open(output, "w") as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print(f"Saved: {output}")


def save_csv(results):

    output = RESULTS_DIR / "benchmark_summary.csv"

    if not results:
        return

    fieldnames = list(results[0].keys())

    with open(
        output,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"Saved: {output}")


def print_summary(results):

    print("\n")
    print("=" * 100)
    print("UCI HEART DISEASE BENCHMARK SUMMARY")
    print("=" * 100)

    print(
        f"{'Dataset':<15}"
        f"{'Model':<20}"
        f"{'CV ROC-AUC':<20}"
        f"{'Test ROC-AUC':<15}"
        f"{'Test F1':<12}"
    )

    print("-" * 100)

    for row in results:

        cv_auc = (
            f"{row['cv_roc_auc_mean']:.4f} "
            f"+/- "
            f"{row['cv_roc_auc_std']:.4f}"
        )

        print(
            f"{row['dataset']:<15}"
            f"{row['selected_model']:<20}"
            f"{cv_auc:<20}"
            f"{row['test_roc_auc']:<15.4f}"
            f"{row['test_f1']:<12.4f}"
        )

    print("=" * 100)


def main():

    results = load_results()

    if not results:
        raise RuntimeError(
            "No benchmark result files found."
        )

    save_json(results)
    save_csv(results)
    print_summary(results)


if __name__ == "__main__":
    main()