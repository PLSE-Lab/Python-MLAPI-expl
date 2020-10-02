import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

conn = sqlite3.connect("../input/database.sqlite")

stats = pd.read_sql_query("select case when r = '' then 0 else r end r, case when sb = '' then 0 else sb end sb from batting", conn)

x = stats['r']
y = stats['sb']

plt.axis([0, x.max() + 10, 0, y.max() + 10])
plt.ylabel('stolen bases')
plt.xlabel('runs')
plt.title("Runs vs. Stolen Bases (all players, all time)")

plt.scatter(x, y)
plt.savefig('runs-vs-stolenbases.png')