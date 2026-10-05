"""
UCI Heart Disease data loading and preprocessing utilities.
"""

from pathlib import Path

import pandas as pd

from .config import FEATURE_NAMES, TARGET_NAME


COLUMN_NAMES = FEATURE_NAMES + [TARGET_NAME]


def load_dataset(path: Path) -> pd.DataFrame:
    """
    Load one of the processed UCI Heart Disease datasets.

    The processed files are comma-separated and use '?' for missing values.
    """

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(
        path,
        header=None,
        names=COLUMN_NAMES,
        na_values=["?"],
    )

    # Convert everything to numeric.
    for column in COLUMN_NAMES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # UCI target:
    # 0 = absence
    # 1-4 = presence/severity
    #
    # We intentionally convert this to binary presence/absence because
    # that is the standard classification task described by UCI.
    df[TARGET_NAME] = (df[TARGET_NAME] > 0).astype(int)

    return df


def dataset_summary(df: pd.DataFrame) -> dict:
    """Return basic dataset statistics."""

    return {
        "rows": int(len(df)),
        "features": len(FEATURE_NAMES),
        "missing_values": int(df[FEATURE_NAMES].isna().sum().sum()),
        "missing_by_feature": {
            feature: int(df[feature].isna().sum())
            for feature in FEATURE_NAMES
            if df[feature].isna().any()
        },
        "class_distribution": {
            str(label): int(count)
            for label, count in df[TARGET_NAME].value_counts().sort_index().items()
        },
    }