#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# # Exploring with a Python Notebook
# 
# The data is available both as a SQLite database and CSV files. We'll work with the SQLite database for this exploration. First, we'll connect to it as follows:

# In[ ]:


import pandas as pd
import sqlite3
con = sqlite3.connect('../input/database.sqlite')


# Let's explore the top 10 competitions by the number of participating teams. To do this, we'll use the Competitions table and the Teams table.

# In[ ]:


print(pd.read_sql_query("""
SELECT c.CompetitionName,
       COUNT(t.Id) NumberOfTeams
FROM Competitions c
INNER JOIN Teams t ON t.CompetitionId=c.Id
-- ONLY including teams that ranked
WHERE t.Ranking IS NOT NULL
GROUP BY c.CompetitionName
ORDER BY COUNT(t.Id) DESC
LIMIT 10
""", con))


# We can look at the top 10 ranked users as follows:

# In[ ]:


top10 = pd.read_sql_query("""
SELECT *
FROM Users
WHERE Ranking IS NOT NULL
ORDER BY Ranking
LIMIT 10
""", con)
print(top10)


# In a similar vein, we can see all the users who ever were ranked one on Kaggle

# In[ ]:


print(pd.read_sql_query("""
SELECT *
FROM Users
WHERE HighestRanking=1
""", con))


# Notebooks are great for showing visualizations as well. Here's a bar plot of the points the top 10 users have:

# In[ ]:


import matplotlib
matplotlib.style.use('ggplot')

top10.sort(columns="Points").plot(x="DisplayName", y="Points", kind="barh", color="#20beff")

