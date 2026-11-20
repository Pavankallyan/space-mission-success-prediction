# Space Mission Success Prediction

Machine learning classification project predicting space mission outcomes.

## Overview
Random Forest classifier (Scikit-Learn) trained on historical space mission data to
predict whether a mission succeeds or fails.

## Approach
- Exploratory analysis with Pandas and Seaborn (`Pictures/` — launch trends, success
  rates by year/company/country, distributions)
- Feature engineering on mission, vehicle, and company attributes
- Model comparison: Logistic Regression vs Random Forest
  (`Pictures/confusion_matrix_*.png`, `roc_curves_models.png`,
  `model_comparison_metrics.png`, `feature_importance_random_forest.png`)
- Hyperparameter tuning with k-fold cross-validation

## Results
- **Random Forest accuracy: 87%** after tuning
- Full report: `Report/Space Mission Success Prediction.pdf`

## Project structure
```
├── Codes/      # Space_Mission_Success_Prediction.ipynb
├── Datasets/   # original, cleaned, and modeling datasets (CSV)
├── Pictures/   # EDA charts, confusion matrices, ROC curves, feature importance
├── Report/     # written report (.pdf + .docx)
├── examples/   # runnable demo of the training pipeline
├── src/        # reusable training pipeline
└── tests/      # pytest unit tests for the pipeline
```

## How to run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook Codes/Space_Mission_Success_Prediction.ipynb
```

## Tests and CI
The reusable training pipeline in `src/pipeline.py` is covered by pytest
unit tests in `tests/` (11 tests, synthetic data mirroring the modeling CSV
schema). GitHub Actions (`.github/workflows/ci.yml`) byte-compiles the sources
and runs the test suite on every push and pull request to `main`:

```bash
pip install -r requirements.txt
python -m pytest -q
```

## Examples
`examples/run_pipeline_demo.py` runs the reusable pipeline on the real
modeling dataset and prints a model comparison, the best model's confusion
matrix, and the random forest's feature importances:

```bash
python examples/run_pipeline_demo.py
```

On the committed dataset (4,324 missions, 1957-2020, 89.7% success rate)
it reports logistic regression F1 0.9458 vs random forest F1 0.9410; the
random forest ranks `year` (0.52) and `loc_city` (0.24) as the most
important features.

## Make shortcuts
The `Makefile` wraps the common commands (`PY` is overridable):

```bash
make setup   # pip install -r requirements.txt
make test    # python -m pytest -q
make demo    # python examples/run_pipeline_demo.py
make clean   # remove caches and bytecode
```
