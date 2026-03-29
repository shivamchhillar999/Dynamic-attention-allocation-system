# Project Report

## Dynamic Attention Allocation System

Latest update 2026-03-28

### Project objective

This project delivers an automated task prioritization model that predicts task priority as `LOW`, `MEDIUM`, or `HIGH` based on a compact set of task attributes.

### Source code structure

- `src/train.py` : data creation, feature engineering, model training, evaluation, and persistence.
- `src/predict.py` : predicts in two modes (single and batch) using the trained model.
- `src/pipeline.py` : orchestrates full workflow end-to-end for reproducible runs.
- `src/data.py` : optional module for dataset generation (if used).
- `src/features.py` : optional module for shared feature calculation logic (if used).

### Data and model artifacts

- `data/data.csv` : generated dataset (5,000 rows by default) with columns:
  - `urgency`, `difficulty`, `time_left`, `importance`, `past_delay`, `priority`
- `model/model.pkl` : trained scikit-learn RandomForest model and label mapping.

### Feature engineering (applied at train and inference)

- `risk = urgency * importance`
- `stress = difficulty * (1 + past_delay)`
- `urgency_adjusted = urgency + 1.2 * past_delay`
- `time_pressure = ((168 - time_left) / 168) * 10` clipped to [0, 10]

### Training procedure

- Stratified 80/20 split to preserve label mix
- RandomForest with 250 trees, `class_weight='balanced'`
- 5-fold cross-validation
- Performance (sample run):
  - CV accuracy: 0.913
  - Test accuracy: 0.919
  - Confusion matrix:
    - LOW: 308/330 correct
    - MEDIUM: 291/330 correct
    - HIGH: 320/340 correct

### Prediction procedure

- Single mode: argument or interactive input
- Batch mode: input CSV + output CSV
- Output includes `predicted_priority` and sorted priority order

### Validation run commands

- `python src/train.py --rebuild`
- `python src/predict.py --mode single --urgency 8 --difficulty 5 --time_left 10 --importance 9 --past_delay 1`
- `python src/predict.py --mode batch --input data/data.csv --output data/results.csv`

### Cleanup and delivery state

- Repository contains requested final structure + documentation
- Temporary process artifacts removed
- One-click reproducibility ensured via `src/pipeline.py`

### Recommendations

- Add regression tests in `tests/`
- Add automated CI validation
- Add model explainability summary (SHAP or feature importances)
- Add data quality checks on `data/data.csv` (value ranges, null handling)
