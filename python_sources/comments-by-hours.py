import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

N = 24
sql_conn = sqlite3.connect('../input/database.sqlite')
hours = []
for i in range(1, N+1):
    val = str(i)
    if i < 10:
        val = '0'+str(i)
    # df = pd.Series(random.randint(10, 40))
    df = pd.read_sql("SELECT COUNT(*) as cnt FROM May2015 WHERE strftime('%H', `created_utc`) = '" + val + "'", sql_conn)
    hours.append(df['cnt'].values[0])
print(hours)

ind = np.arange(1, N+1, 1)
width = 0.5
fig, ax = plt.subplots(figsize=(10,5))
p1 = plt.bar(ind, hours, width, color='r')
plt.xlabel('Hour of day')
plt.ylabel('Number of comments')
plt.title('Number of comments by hours')
plt.xticks(ind + width/2., np.arange(1, N+1, 1))
plt.yticks(np.arange(0, max(hours)+2, round((max(hours)+2)/10 + 1, 0)))
plt.xlim(0, N+1)
plt.ylim(min(hours)-1, max(hours)+1)

plt.show()
plt.savefig("comments_by_hours.png")
