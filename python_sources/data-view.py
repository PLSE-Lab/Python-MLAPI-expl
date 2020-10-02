import sqlite3
import pandas as pd

sql_con = sqlite3.connect('../input/database.sqlite')

sql_cmd = "select * from May2015 limit 1000000"

data = pd.read_sql(sql_cmd, sql_con)

print(data.describe())