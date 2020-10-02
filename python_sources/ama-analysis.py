import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import pylab 

fig = plt.figure()

db = '../input/database.sqlite'
conn = sqlite3.connect(db)
conn.execute('PRAGMA case_sensitive_like=ON;')
df = pd.read_sql(("SELECT * FROM May2015 \
where body like '% AMA %' \
ORDER BY created_utc limit 20"),conn)

for text in df['body']:
    print(text)
