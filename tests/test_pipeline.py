"""Unit tests for src/pipeline.py."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pipeline import (
    TARGET_COLUMN,
    evaluate,
    load_modeling_dataset,
    prepare_features,
    run_pipeline,
    split_dataset,
    train_logistic_regression,
    train_random_forest,
)


def test_prepare_features_drops_target(synthetic_missions):
    X, y = prepare_features(synthetic_missions)
    assert TARGET_COLUMN not in X.columns
    assert list(y) == list(synthetic_missions[TARGET_COLUMN])


def test_prepare_features_fills_nan_with_median(synthetic_missions):
    median_year = synthetic_missions["year"].median()
    X, _ = prepare_features(synthetic_missions)
    assert X.isna().sum().sum() == 0
    nan_rows = synthetic_missions["year"].isna()
    assert (X.loc[nan_rows, "year"] == median_year).all()


def test_prepare_features_does_not_mutate_input(synthetic_missions):
    before = synthetic_missions.copy()
    prepare_features(synthetic_missions)
    pd.testing.assert_frame_equal(synthetic_missions, before)


def test_load_modeling_dataset_roundtrip(synthetic_missions, tmp_path):
    csv = tmp_path / "modeling_dataset.csv"
    synthetic_missions.to_csv(csv, index=False)
    loaded = load_modeling_dataset(str(csv))
    pd.testing.assert_frame_equal(loaded, synthetic_missions)


def test_split_dataset_shapes_and_stratification(synthetic_missions):
    X_train, X_test, y_train, y_test = split_dataset(synthetic_missions)
    assert len(X_train) == 64  # 80 * 0.8
    assert len(X_test) == 16
    # both classes present in both splits thanks to stratification
    assert set(y_train.unique()) == {0, 1}
    assert set(y_test.unique()) == {0, 1}
    # features never see the target column
    assert TARGET_COLUMN not in X_train.columns


def test_split_dataset_deterministic(synthetic_missions):
    first = split_dataset(synthetic_missions)
    second = split_dataset(synthetic_missions)
    pd.testing.assert_frame_equal(first[0], second[0])


def test_train_random_forest_fits_and_predicts(synthetic_missions):
    X_train, X_test, y_train, y_test = split_dataset(synthetic_missions)
    model = train_random_forest(X_train, y_train)
    preds = model.predict(X_test)
    assert len(preds) == len(y_test)
    assert set(np.unique(preds)) <= {0, 1}


def test_train_logistic_regression_fits_and_predicts(synthetic_missions):
    X_train, X_test, y_train, y_test = split_dataset(synthetic_missions)
    model = train_logistic_regression(X_train, y_train)
    preds = model.predict(X_test)
    assert len(preds) == len(y_test)
    assert set(np.unique(preds)) <= {0, 1}


def test_evaluate_reports_all_metrics(synthetic_missions):
    X_train, X_test, y_train, y_test = split_dataset(synthetic_missions)
    model = train_random_forest(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    assert set(metrics) == {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "confusion_matrix",
    }
    for key in ("accuracy", "precision", "recall", "f1"):
        assert 0.0 <= metrics[key] <= 1.0
    assert sum(sum(row) for row in metrics["confusion_matrix"]) == len(y_test)


def test_evaluate_perfect_predictions(synthetic_missions):
    X, y = prepare_features(synthetic_missions)

    class Oracle:
        def predict(self, X):
            return np.asarray(y)

    metrics = evaluate(Oracle(), X, y)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)


def test_run_pipeline_end_to_end(synthetic_missions, tmp_path):
    csv = tmp_path / "modeling_dataset.csv"
    synthetic_missions.to_csv(csv, index=False)
    summary = run_pipeline(dataset_path=str(csv))
    assert set(summary) == {"logistic_regression", "random_forest"}
    for metrics in summary.values():
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert metrics["confusion_matrix"]
