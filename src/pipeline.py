"""Reusable training pipeline for space mission success prediction.

Wraps the modeling flow from ``Codes/Space_Mission_Success_Prediction.ipynb`` in
plain functions so it can be re-run without a notebook:

1. :func:`load_modeling_dataset` — read the encoded modeling CSV
2. :func:`split_dataset` — stratified train/test split on ``mission_success``
3. :func:`train_random_forest` / :func:`train_logistic_regression` — fit a model
4. :func:`evaluate` — accuracy, precision, recall, F1 and the confusion matrix
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "mission_success"
DEFAULT_TEST_SIZE = 0.2
RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "company",
    "loc_city",
    "loc_state",
    "loc_country",
    "rocket",
    "year",
    "status_rocket",
]


def load_modeling_dataset(path: str = "Datasets/modeling_dataset.csv") -> pd.DataFrame:
    """Load the encoded modeling dataset.

    Parameters
    ----------
    path:
        Path to the modeling CSV (defaults to the repo-relative location).

    Returns
    -------
    pd.DataFrame
        One row per mission with the encoded features and the
        ``mission_success`` target column.
    """
    return pd.read_csv(path)


def prepare_features(
    df: pd.DataFrame, target_column: str = TARGET_COLUMN
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a loaded dataset into features and target, handling missing values.

    Numeric features with missing values (e.g. ``year``) are filled with the
    column median, since the sklearn models below do not accept NaN inputs.

    Returns
    -------
    tuple
        ``(X, y)`` — feature frame and target series.
    """
    X = df.drop(columns=[target_column]).copy()
    for col in X.select_dtypes(include="number").columns:
        X[col] = X[col].fillna(X[col].median())
    return X, df[target_column]


def split_dataset(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """Stratified train/test split on the ``mission_success`` target.

    Missing feature values are filled with column medians via
    :func:`prepare_features` before splitting.

    Returns
    -------
    tuple
        ``(X_train, X_test, y_train, y_test)``.
    """
    X, y = prepare_features(df, target_column=target_column)
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def train_random_forest(
    X_train: pd.DataFrame, y_train: pd.Series, random_state: int = RANDOM_STATE
) -> RandomForestClassifier:
    """Fit a RandomForestClassifier on the training split."""
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(
    X_train: pd.DataFrame, y_train: pd.Series, random_state: int = RANDOM_STATE
) -> LogisticRegression:
    """Fit a LogisticRegression on the training split."""
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Score a fitted model on the test split.

    Returns
    -------
    dict
        ``accuracy``, ``precision``, ``recall``, ``f1`` and ``confusion_matrix``
        (as a nested list).
    """
    preds = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
    }


def run_pipeline(
    dataset_path: str = "Datasets/modeling_dataset.csv", random_state: int = RANDOM_STATE
) -> dict:
    """End-to-end run: load, split, train both models, evaluate.

    Returns
    -------
    dict
        ``{"logistic_regression": <metrics>, "random_forest": <metrics>}``.
    """
    df = load_modeling_dataset(dataset_path)
    X_train, X_test, y_train, y_test = split_dataset(df, random_state=random_state)
    results = {}
    for name, trainer in (
        ("logistic_regression", train_logistic_regression),
        ("random_forest", train_random_forest),
    ):
        model = trainer(X_train, y_train, random_state=random_state)
        results[name] = evaluate(model, X_test, y_test)
    return results


if __name__ == "__main__":  # pragma: no cover
    import json

    summary = run_pipeline()
    print(
        json.dumps(
            {
                name: {k: v for k, v in metrics.items() if k != "confusion_matrix"}
                for name, metrics in summary.items()
            },
            indent=2,
        )
    )
