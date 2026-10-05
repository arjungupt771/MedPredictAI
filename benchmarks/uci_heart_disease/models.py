"""
Model definitions for the UCI Heart Disease benchmark.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from .config import (
    CATEGORICAL_FEATURES,
    FEATURE_NAMES,
    NUMERIC_FEATURES,
    RANDOM_SEED,
)


def build_preprocessor(
    scale_numeric: bool = True,
    feature_names=None,
):
    """
    Build preprocessing pipeline.

    Numeric:
        median imputation + optional standardization

    Categorical:
        most-frequent imputation + one-hot encoding
    """

    feature_names = FEATURE_NAMES if feature_names is None else feature_names
    numeric_features = [
        feature
        for feature in NUMERIC_FEATURES
        if feature in feature_names
    ]
    categorical_features = [
        feature
        for feature in CATEGORICAL_FEATURES
        if feature in feature_names
    ]

    numeric_steps = [
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                keep_empty_features=True,
            ),
        ),
    ]

    if scale_numeric:
        numeric_steps.append(
            ("scaler", StandardScaler())
        )

    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent",
                keep_empty_features=True,
            ),
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ])

    return ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            numeric_features,
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features,
        ),
    ])


def build_models(feature_names=None):
    """Return all benchmark models."""

    feature_names = FEATURE_NAMES if feature_names is None else feature_names
    models = {}

    models["LogisticRegression"] = Pipeline([
        (
            "preprocessor",
            build_preprocessor(
                scale_numeric=True,
                feature_names=feature_names,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_SEED,
            ),
        ),
    ])

    models["RandomForest"] = Pipeline([
        (
            "preprocessor",
            build_preprocessor(
                scale_numeric=False,
                feature_names=feature_names,
            ),
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=2,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),
        ),
    ])

    models["XGBoost"] = Pipeline([
        (
            "preprocessor",
            build_preprocessor(
                scale_numeric=False,
                feature_names=feature_names,
            ),
        ),
        (
            "classifier",
            XGBClassifier(
                n_estimators=250,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),
        ),
    ])

    return models