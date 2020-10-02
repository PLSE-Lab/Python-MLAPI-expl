import sqlite3
import pandas as pd
import re
import matplotlib.pyplot as plt

sql_conn = sqlite3.connect('../input/database.sqlite')

Users = pd.read_sql("SELECT author, author_flair_text FROM May2015 WHERE subreddit = 'politics' AND author_flair_text <> 'None'", sql_conn)
Users.drop_duplicates(inplace = True)

Comments = pd.read_sql("SELECT body, author, gilded, edited, controversiality FROM May2015 WHERE subreddit = 'politics'", sql_conn)
print(Comments.info())