#!/usr/bin/env python
# coding: utf-8

# # Kaggle Hall of Fame
# 
# Users who achieved a [sitewide #1 ranking](Would also like to maybe squeeze in a couple ops updates) for their Kaggle Competition performance.

# In[ ]:


from IPython.display import HTML
import pandas as pd
import re
import sqlite3

con = sqlite3.connect('../input/database.sqlite')

users = pd.read_sql_query("""
SELECT u.Id UserId,
       u.DisplayName,
       u.HighestRanking HighestRanking,
       u.Ranking CurrentRanking
FROM Users u
WHERE u.HighestRanking=1
ORDER BY u.Ranking""", con)

users["User"] = ""
users["HighestRanking"] = users["HighestRanking"].astype("int")
users["CurrentRanking"] = users["CurrentRanking"].astype("int")

for i in range(len(users)):
    users.loc[i, "User"] = "<" + "a href='https://www.kaggle.com/u/" + str(users["UserId"][i]) + "'>" + users["DisplayName"][i] + "<" + "/a>"

users.index = range(1, len(users)+1)
pd.set_option("display.max_colwidth", -1)

HTML(users[["User", "HighestRanking", "CurrentRanking"]].to_html(escape=False))

