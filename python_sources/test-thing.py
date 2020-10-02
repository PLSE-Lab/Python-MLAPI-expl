import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sql_conn = sqlite3.connect('../input/database.sqlite')

# There is so much data here I hit the 8gb memory limit if I try and grab it all
sql_cmd = "SELECT subreddit, body, score FROM May2015 WHERE lower(body) LIKE '%Yes%' OR lower(body) LIKE '%No%' ORDER BY score"

data = pd.read_sql(sql_cmd, sql_conn)