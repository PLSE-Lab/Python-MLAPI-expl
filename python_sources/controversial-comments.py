import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sql_conn = sqlite3.connect('../input/database.sqlite')

# There is so much data here I hit the 8gb memory limit if I try and grab it all
sql_cmd = "Select COUNT(*) From May2015" #

num_of_comments = pd.read_sql(sql_cmd, sql_conn)

sql_cmd = "Select COUNT(*) From May2015 WHERE controversiality=1" #
num_of_controversial = pd.read_sql(sql_cmd, sql_conn)

# Find the percentage of controversial comments
print("Number of controversial comments: %s" % num_of_controversial)
print(1.0 * num_of_controversial / num_of_comments)