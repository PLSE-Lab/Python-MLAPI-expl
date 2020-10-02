import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sql_conn = sqlite3.connect('../input/database.sqlite')

#sql_cmd = "Select subreddit, controversiality From May2015 ORDER BY Random() LIMIT 5000000" #
#sql_cmd = "SELECT * FROM May2015 WHERE subreddit = \"SquaredCircle\"" #
sql_cmd = 'SELECT subreddit FROM May2015 GROUP BY subreddit HAVING COUNT(DISTINCT(author)) > 3'
data = pd.read_sql(sql_cmd, sql_conn)
print(len(data))

#data['body'] = data.body.apply(lambda s: s.encode('ascii','ignore'))
#data.to_csv('sc.csv')