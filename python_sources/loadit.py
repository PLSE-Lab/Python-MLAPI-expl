import sqlite3
import pandas as pd

sql_conn = sqlite3.connect('../input/database.sqlite')

res = pd.read_sql("SELECT body, score FROM May2015 WHERE LENGTH(body) < 51 AND LENGTH(body) > 10 AND body LIKE 'http%.gif%' ORDER BY score DESC LIMIT 100", sql_conn)

for comment in res['body']:
    print(comment)