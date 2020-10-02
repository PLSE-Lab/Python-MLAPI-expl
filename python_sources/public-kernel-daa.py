import pandas as pd
import os

if os.path.exists('./train.csv'):
    df = pd.read_csv('./train.csv')
    predicted = pd.DataFrame()
    df['day_of_week'] = df.visits.apply(lambda x: [(y - 1) % 7 + 1 for y in x][-50:])
    predicted['id'] = range(1, len(df) + 1)
    predicted['nextvisit'] = df.day_of_week.apply(lambda x: max([(len([z for z in x if z == y]), y) for y in set(x)])[1])
    predicted.to_csv('solution.csv', index=False)
    