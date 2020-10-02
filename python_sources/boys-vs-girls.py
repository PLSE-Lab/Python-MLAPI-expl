import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

conn = sqlite3.connect('../input/database.sqlite')
cursor = conn.cursor()

results = cursor.execute("SELECT Year, Sum(CASE WHEN Gender = 'F' THEN Count END) *1.0 / SUM(Count) - SUM(CASE WHEN Gender = 'M' THEN Count END) *1.0 / SUM(Count) AS DeltaRatio FROM NationalNames GROUP BY Year")

fig = plt.figure(figsize=(12,9))

a = np.fromiter(results.fetchall(), dtype=('i4,f8'))

x_arr = np.array([x[1]*100 for x in a])
y_arr = np.array([x[0] for x in a])

ax = plt.gca()

ax.set_yticks(np.arange(1880,2014,10), minor=False)
ax.yaxis.grid(True, which='major')

ax.set_xticks(np.arange(-100,100,10), minor=False)
ax.xaxis.grid(True, which='major')

plt.title('Percentage of Births: Boys (-) vs. Girls (+)')

plt.hlines(y_arr, 0, x_arr)
plt.plot([0,0], [y_arr.min(), y_arr.max()], '--')

plt.savefig('boys-v-girls.png')