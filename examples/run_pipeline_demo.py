"""End-to-end demo of the space mission success pipeline.

Runs the reusable training pipeline in ``src/pipeline.py`` on the real
modeling dataset (``Datasets/modeling_dataset.csv``) and prints a readable
model comparison: per-model accuracy / precision / recall / F1, confusion
matrices, the best model by F1, and the random forest's feature importances.

Usage (from the repo root)::

    pip install -r requirements.txt
    python examples/run_pipeline_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pipeline import (  # noqa: E402
    evaluate,
    load_modeling_dataset,
    split_dataset,
    train_logistic_regression,
    train_random_forest,
)


def _metric_table(results: dict) -> str:
    """Render the per-model metrics as a plain-text table."""
    header = f"{'model':<22}{'accuracy':>10}{'precision':>11}{'recall':>9}{'f1':>9}"
    rows = [header, "-" * len(header)]
    for name, metrics in results.items():
        rows.append(
            f"{name:<22}"
            f"{metrics['accuracy']:>10.4f}"
            f"{metrics['precision']:>11.4f}"
            f"{metrics['recall']:>9.4f}"
            f"{metrics['f1']:>9.4f}"
        )
    return "\n".join(rows)


def _confusion_matrix(cm: list) -> str:
    """Render a 2x2 confusion matrix with predicted/actual labels."""
    return (
        "                 predicted 0   predicted 1\n"
        f"  actual 0     {cm[0][0]:>10}  {cm[0][1]:>10}\n"
        f"  actual 1     {cm[1][0]:>10}  {cm[1][1]:>10}"
    )


def main() -> int:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_path = os.path.join(repo_root, "Datasets", "modeling_dataset.csv")

    df = load_modeling_dataset(dataset_path)
    print(f"Dataset: {len(df)} missions, {len(df.columns)} columns")
    print(f"Success rate: {df['mission_success'].mean():.1%}")
    print(f"Year range: {int(df['year'].min())}-{int(df['year'].max())}")
    print()

    X_train, X_test, y_train, y_test = split_dataset(df)
    print(f"Train: {len(X_train)} missions, test: {len(X_test)} missions")
    print()

    models = {
        "logistic_regression": train_logistic_regression(X_train, y_train),
        "random_forest": train_random_forest(X_train, y_train),
    }
    results = {name: evaluate(model, X_test, y_test) for name, model in models.items()}

    print("Model comparison (test set)")
    print(_metric_table(results))
    print()

    best = max(results, key=lambda name: results[name]["f1"])
    print(f"Best model by F1: {best} (F1 = {results[best]['f1']:.4f})")
    print()
    print(f"Confusion matrix — {best}")
    print(_confusion_matrix(results[best]["confusion_matrix"]))
    print()

    importances = sorted(
        zip(X_train.columns, models["random_forest"].feature_importances_),
        key=lambda pair: pair[1],
        reverse=True,
    )
    print("Random forest feature importances")
    for feature, importance in importances:
        print(f"  {feature:<15}{importance:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
