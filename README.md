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
└── Report/     # written report (.pdf + .docx)
```

## How to run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook Codes/Space_Mission_Success_Prediction.ipynb
```
