import argparse
import os

import joblib
import numpy as np
import pandas as pd

FEATURE_COLUMNS = ['urgency', 'difficulty', 'time_left', 'importance', 'past_delay']


def build_features(df):
    feat = df[FEATURE_COLUMNS].copy()
    feat['risk'] = feat['urgency'] * feat['importance']
    feat['stress'] = feat['difficulty'] * (1 + feat['past_delay'])
    feat['urgency_adjusted'] = feat['urgency'] + 1.2 * feat['past_delay']
    feat['time_pressure'] = np.clip((168 - feat['time_left']) / 168 * 10, 0, 10)
    return feat


def load_model(model_path='model/model.pkl'):
    if not os.path.isfile(model_path):
        raise FileNotFoundError('Model file not found: ' + model_path)
    data = joblib.load(model_path)
    return data['model'], data.get('mapping', {0: 'LOW', 1: 'MEDIUM', 2: 'HIGH'})


def predict_priority(model, urgency, difficulty, time_left, importance, past_delay):
    row = pd.DataFrame([[urgency, difficulty, time_left, importance, past_delay]], columns=FEATURE_COLUMNS)
    features = build_features(row)
    pred = model.predict(features)[0]
    return {0: 'LOW', 1: 'MEDIUM', 2: 'HIGH'}.get(pred, 'UNKNOWN')


def predict_batch(model, input_path='data/data.csv', output_path='data/results.csv'):
    table = pd.read_csv(input_path)
    required = FEATURE_COLUMNS
    if not all(col in table.columns for col in required):
        raise ValueError('CSV must contain: ' + ', '.join(required))
    features = build_features(table)
    pred = model.predict(features)
    table['predicted_priority'] = [{0:'LOW',1:'MEDIUM',2:'HIGH'}.get(v,'UNKNOWN') for v in pred]
    table['priority_score'] = pred
    table = table.sort_values(['priority_score', 'importance', 'urgency'], ascending=[False, False, False])
    table = table.drop(columns=['priority_score'])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    table.to_csv(output_path, index=False)
    return table


def main():
    parser = argparse.ArgumentParser(description='Predict priority')
    parser.add_argument('--mode', choices=['single', 'batch'], default='single')
    parser.add_argument('--model', default='model/model.pkl')
    parser.add_argument('--input', default='data/data.csv')
    parser.add_argument('--output', default='data/results.csv')
    parser.add_argument('--urgency', type=float, default=None)
    parser.add_argument('--difficulty', type=float, default=None)
    parser.add_argument('--time_left', type=float, default=None)
    parser.add_argument('--importance', type=float, default=None)
    parser.add_argument('--past_delay', type=int, choices=[0, 1], default=None)
    args = parser.parse_args()
    model, mapping = load_model(args.model)
    if args.mode == 'single':
        if args.urgency is None or args.difficulty is None or args.time_left is None or args.importance is None or args.past_delay is None:
            args.urgency = float(input('urgency 0-10: ').strip())
            args.difficulty = float(input('difficulty 0-10: ').strip())
            args.time_left = float(input('time_left hours: ').strip())
            args.importance = float(input('importance 0-10: ').strip())
            args.past_delay = int(input('past_delay 0 or 1: ').strip())
        print('predicted', predict_priority(model, args.urgency, args.difficulty, args.time_left, args.importance, args.past_delay))
    else:
        table = predict_batch(model, input_path=args.input, output_path=args.output)
        print(table.head(20))


if __name__ == '__main__':
    main()
