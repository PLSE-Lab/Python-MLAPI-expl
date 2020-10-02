import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np; 

PI = np.pi
I = np.complex(0,1)

TALKY = True
def log(s):
    if TALKY:
        print(s)

SECS_PER_DAY = 24*60*60.


sql_conn = sqlite3.connect('../input/database.sqlite')
log('connected to db')

df = pd.read_sql("SELECT author, created_utc FROM May2015 ORDER BY author LIMIT 5000000 ", sql_conn)
log('got df from db')

df = df.groupby('author').filter(lambda x: len(x)>25)

phase = PI * 2 * (df['created_utc'] % SECS_PER_DAY) / SECS_PER_DAY
df['fourval'] = np.exp(I * phase)

authors = df.groupby('author')

component = authors.mean().abs()
component['counts'] = authors['author'].count()

morethan = component[component['counts']>0]

print(morethan.describe())
print(morethan.sort('fourval'))