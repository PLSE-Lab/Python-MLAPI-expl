import sqlite3
import pandas as pd
import re
import matplotlib.pyplot as plt

sql_conn = sqlite3.connect('../input/database.sqlite')

Users = pd.read_sql("SELECT * FROM May2015 WHERE body LIKE '%Waterloo%' or body LIKE '%waterloo%'", sql_conn)
print(Users.shape)



