import sqlite3
import sys
import os
import pandas as pd

import codecs

print(sys.version)

conn = sqlite3.connect('../input/database.sqlite')
#cmd = "Select * From May2015 WHERE subreddit == 'newzealand' LIMIT 1000"
cmd = "Select * FROM May2015 group by subreddit ORDER BY COUNT(subreddit) LIMIT 10000"

df = pd.read_sql(cmd, conn)

body_utf8 = [codecs.encode(i,'utf-8') for i in df.body]
df.body = pd.Series(body_utf8)

for i in range(0, 1000):
    print( df["body"][i], df["subreddit"][i] )