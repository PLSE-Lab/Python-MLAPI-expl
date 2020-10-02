import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sq_conn = sqlite3.connect('../input/database.sqlite')

data = sq_conn.execute("SELECT author, subreddit FROM May2015 WHERE author IN (SELECT DISTINCT author FROM May2015) LIMIT 10000")
print(data)

subs = data.fetchall()

df = pd.DataFrame(subs, columns = ["User", "Subreddit"])
print(df.head())

# remove [deleted] comments
df = df[df["User"] != "[deleted]"]
df = df.sort_values(by = "User")
print(df.head())

users = df["User"].unique()
print(users[:5])

subreddits = df["Subreddit"].unique()
print(subreddits[:5])

sub_n = len(subreddits)
print(sub_n)

matrix_zero = np.zeros((sub_n, sub_n))

adjacency_mat = pd.DataFrame(data = matrix_zero, index = subreddits, columns = subreddits)

for user in users:
    temp = df[df["User"] == user]
    for sub1 in temp["Subreddit"]:
        for sub2 in temp["Subreddit"]:
            adjacency_mat[sub1][sub2] += 1
            
print(adjacency_mat.head())

adjacency_mat.to_csv("all.csv")