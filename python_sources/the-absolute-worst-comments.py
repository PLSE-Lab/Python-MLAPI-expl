import sqlite3
import pandas as pd
import textwrap

sql_conn = sqlite3.connect('../input/database.sqlite')
df = pd.read_sql("SELECT * FROM May2015 ORDER BY gilded DESC LIMIT 100", sql_conn)


df1 = df[df['body'] != '[deleted]']
d = df1[:20]

for i, row in d.iterrows():
    print('Score: {}'.format(row['score']))
    print('gilded: {}'.format(row['gilded']))
    print('Subreddit: {}'.format(row['subreddit']))
    print('Author: {}'.format(row['author']))
    body = '\n\n'.join(textwrap.fill(lines, 80) for lines in row['body'].split('\n\n'))
    print(body)
    print('-----------------------------')
