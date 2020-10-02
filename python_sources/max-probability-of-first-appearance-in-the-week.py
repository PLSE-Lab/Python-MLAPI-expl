from tqdm import tqdm
import pandas as pd
import numpy as np

def save_predictions(predictions, name):
    fname = f'solution{name}.csv'
    with open(fname, 'wt') as f:
        f.write('id,nextvisit\n')
        for i, res in enumerate(predictions):
            f.write(f'{i+1}, {res}\n')
    print(f'Predictions was saved to {fname}')

def get_week(day):
    return day // 7

def get_week_day(day):
    return day % 7

def compute_weekday_probs(days):
    first_day_in_week = dict()
    for day in days:
        week = get_week(day)
        week_day = get_week_day(day)
        if week not in first_day_in_week:
            first_day_in_week[week] = week_day

    weekday_probs = np.zeros(7, dtype=np.float32)
    for week, weekday in first_day_in_week.items():
        weekday_probs[weekday] += 1.0

    weekday_probs /= weekday_probs.sum()
    return np.argmax(weekday_probs) + 1

def main():
    data_path = 'train.csv'
    df = pd.read_csv(data_path)
    predictions = []
    for visits in tqdm(df.visits, desc='Prediction of the next appearance'):
        days = list(map(lambda x: int(x) - 1, visits.split()))
        weekday_probs = compute_weekday_probs(days)
        predictions.append(weekday_probs)

    save_predictions(predictions, '_01')