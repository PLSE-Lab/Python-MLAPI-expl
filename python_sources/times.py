# coding: utf-8

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

N = 31
sql_conn = sqlite3.connect('../input/database.sqlite')
days = []
for i in range(1, N+1):
    val = str(i)
    if i < 10:
        val = '0'+str(i)
    df = pd.read_sql("SELECT COUNT(*) as cnt FROM May2015 WHERE strftime('%d', `created_utc`) = '" + val + "'", sql_conn)
    # df = pd.read_sql("SELECT COUNT(*) as cnt FROM May2015", sql_conn)
    days.append(df['cnt'].values[0])
    print(df)
print(days)

ind = np.arange(1, N+1, 1)
width = 0.5
fig, ax = plt.subplots(figsize=(10,5))
p1 = plt.bar(ind, days, width, color='r')
plt.xlabel('Day of month')
plt.ylabel('Number of comments')
plt.title('Number of comments by days')
plt.xticks(ind + width/2., np.arange(1, N+1, 1))
plt.yticks(np.arange(0, max(days)+2, round((max(days)+2)/10 + 1, 0)))
plt.xlim(0, N+1)
plt.ylim(min(days)-1, max(days)+1)

plt.show()
plt.savefig("comments_by_days.png")
