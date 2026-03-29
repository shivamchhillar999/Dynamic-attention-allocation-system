import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split

DATA_PATH = os.path.join('data', 'data.csv')
MODEL_PATH = os.path.join('model', 'model.pkl')

FEATURE_COLUMNS = ['urgency', 'difficulty', 'time_left', 'importance', 'past_delay']


def build_synthetic_data(path=DATA_PATH, rows=5000, seed=42):
    rng = np.random.default_rng(seed)
    table = pd.DataFrame({
        'urgency': rng.uniform(0, 10, rows),
        'difficulty': rng.uniform(0, 10, rows),
        'time_left': rng.uniform(0.1, 168, rows),
        'importance': rng.uniform(0, 10, rows),
        'past_delay': rng.integers(0, 2, rows),
    })
    urgency_factor = table['urgency'] ** 1.1
    importance_factor = table['importance'] ** 1.05
    delay_factor = 5 * table['past_delay']
    time_factor = np.clip((168 - table['time_left']) / 168 * 10, 0, 10)
    score = 0.35 * urgency_factor + 0.30 * importance_factor + 0.20 * delay_factor + 0.15 * time_factor - 0.25 * table['difficulty']
    thresholds = np.quantile(score, [0.33, 0.66])
    table['priority'] = np.where(score >= thresholds[1], 'HIGH', np.where(score >= thresholds[0], 'MEDIUM', 'LOW'))
    table['urgency'] = np.clip(table['urgency'] + rng.normal(0, 0.2, rows), 0, 10)
    table['difficulty'] = np.clip(table['difficulty'] + rng.normal(0, 0.2, rows), 0, 10)
    table['importance'] = np.clip(table['importance'] + rng.normal(0, 0.2, rows), 0, 10)
    table['time_left'] = np.clip(table['time_left'] + rng.normal(0, 2.0, rows), 0.1, 168)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    table.to_csv(path, index=False)


def build_features(df):
    feat = df[FEATURE_COLUMNS].copy()
    feat['risk'] = feat['urgency'] * feat['importance']
    feat['stress'] = feat['difficulty'] * (1 + feat['past_delay'])
    feat['urgency_adjusted'] = feat['urgency'] + 1.2 * feat['past_delay']
    feat['time_pressure'] = np.clip((168 - feat['time_left']) / 168 * 10, 0, 10)
    return feat


def train_model(data_path=DATA_PATH, model_path=MODEL_PATH, min_rows=1000, rows=5000, seed=42, rebuild=False):
    if rebuild or not os.path.isfile(data_path):
        build_synthetic_data(path=data_path, rows=rows, seed=seed)
    data = pd.read_csv(data_path)
    required = FEATURE_COLUMNS + ['priority']
    if not all(col in data.columns for col in required):
        raise ValueError('data.csv must contain ' + ', '.join(required))
    if len(data) < min_rows:
        build_synthetic_data(path=data_path, rows=rows, seed=seed)
        data = pd.read_csv(data_path)
    data['priority'] = data['priority'].astype(str).str.upper().str.strip()
    mapping = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}
    data['priority'] = data['priority'].map(mapping)
    if data['priority'].isna().any():
        raise ValueError('priority column must contain LOW/MEDIUM/HIGH')
    X = build_features(data)
    y = data['priority']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    model = RandomForestClassifier(n_estimators=250, class_weight='balanced', random_state=seed)
    cv_scores = cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy')
    print('cross validation:', cv_scores.mean().round(4), cv_scores)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print('accuracy:', round(accuracy_score(y_test, y_pred), 4))
    print(classification_report(y_test, y_pred, target_names=['LOW', 'MEDIUM', 'HIGH'], zero_division=0))
    print(confusion_matrix(y_test, y_pred))
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({'model': model, 'mapping': mapping}, model_path)


def main():
    parser = argparse.ArgumentParser(description='Train Dynamic Attention model')
    parser.add_argument('--data', default=DATA_PATH)
    parser.add_argument('--model', default=MODEL_PATH)
    parser.add_argument('--min_rows', type=int, default=1000)
    parser.add_argument('--rows', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rebuild', action='store_true')
    args = parser.parse_args()
    train_model(data_path=args.data, model_path=args.model, min_rows=args.min_rows, rows=args.rows, seed=args.seed, rebuild=args.rebuild)


if __name__ == '__main__':
    main()
