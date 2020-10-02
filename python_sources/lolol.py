import sqlite3
import pandas as pd

sql_conn = sqlite3.connect('../input/database.sqlite')
df = pd.read_sql('SELECT * FROM May2015 WHERE SCORE < -100 ORDER BY score ASC', sql_conn)

df1 = df[df['body'] != '[deleted]']
d = df1[:20]

for i, row in d.iterrows():
    print('Score: {}'.format(row['score']))
    print('Subreddit: {}'.format(row['subreddit']))
    print('Author: {}'.format(row['author']))
    print(row['body'])
    print('-----------------------------')