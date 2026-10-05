"""
MedPredict AI - Lightweight Experiment Tracker
Logs one JSON file per training run to experiments/. Upgrade path to MLflow
later: this schema maps directly onto an MLflow run's params/metrics/artifacts.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).parent.parent / "experiments"
EXPERIMENTS_DIR.mkdir(exist_ok=True)


def log_experiment(name: str, hyperparameters: dict, metrics: dict, notes: str = "") -> Path:
    """
    Writes experiments/<name>.json with a timestamp, so re-running an
    experiment overwrites its own file rather than accumulating duplicates —
    keep the run history in git log if you need every historical run.
    """
    record = {
        "name": name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hyperparameters": hyperparameters,
        "metrics": metrics,
        "notes": notes,
    }
    path = EXPERIMENTS_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    return path


def list_experiments() -> list[dict]:
    """Reads back every experiments/*.json, for a leaderboard view."""
    runs = []
    for p in sorted(EXPERIMENTS_DIR.glob("*.json")):
        with open(p) as f:
            runs.append(json.load(f))
    return runs


def best_experiment(metric: str = "f1_weighted") -> dict | None:
    runs = list_experiments()
    if not runs:
        return None
    return max(runs, key=lambda r: r["metrics"].get(metric, 0))