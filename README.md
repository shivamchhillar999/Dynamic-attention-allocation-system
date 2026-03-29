# Dynamic Attention Allocation System

This repository is a small, polished, and reproducible `Python` project for predicting task priority levels: `LOW`, `MEDIUM`, `HIGH`.

## What this does

- Generate a synthetic dataset (or use your own CSV).
- Train a Random Forest model over task features.
- Predict priorities for single items or batch files.
- Sort tasks by recommended priority order.

## Project structure

Dynamic-Attention-Allocation-System/
├── data/
│   └── data.csv
├── model/
│   └── model.pkl
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── data.py
│   ├── features.py
│   └── pipeline.py
├── README.md
└── requirements.txt

## Setup

1. Create virtual environment

   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   ```

2. Install requirements

   ```bash
   pip install -r requirements.txt
   ```

## Run it

### Full workflow (recommended)

```bash
python src/pipeline.py
```

### Train only

```bash
python src/train.py --data data/data.csv --model model/model.pkl --rebuild
```

### Prediction

#### Single input (interactive fallback)

```bash
python src/predict.py --mode single
```

or values with flags:

```bash
python src/predict.py --mode single --urgency 8 --difficulty 5 --time_left 10 --importance 9 --past_delay 1
```

#### Batch input

```bash
python src/predict.py --mode batch --input data/data.csv --output data/results.csv
```

## Key details

- Base features: `urgency`, `difficulty`, `time_left`, `importance`, `past_delay`
- Derived features: `risk`, `stress`, `urgency_adjusted`, `time_pressure`
- Model: `RandomForestClassifier` with balanced class weights

## What is in `data/` and `model/`

- `data/data.csv`: contains task samples and ground truth priorities (generated if absent)
- `model/model.pkl`: saved trained model (produced by `train.py`)

## Suggested enhancements

- Add unit/integration tests (`tests/`)
- Add GitHub Actions CI for `pytest`
- Add model explainability (e.g., SHAP)
- Add `pyproject.toml` for packaging

## Project report

See `PROJECT_REPORT.md` for the latest summary of what was done and why.
