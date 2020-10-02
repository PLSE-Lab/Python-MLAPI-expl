#!/usr/bin/env python
# coding: utf-8

# ![](https://assets-cdn.github.com/images/modules/open_graph/github-octocat.png)
# # Introduction
# > Hi! In this notebook I will solve SQL Scavenger Hunt Day: Day 5 challenge and do some more analysis Github repos big query dataset. If you find any problem or have any suggestions feel free to reach comment section. And don't forget to upvote! 
# 

# In[ ]:


import numpy as np # linear algebra
# pandas for handling data
import pandas as pd
# google bigquery library for quering data
from google.cloud import bigquery
# BigQueryHelper for converting query result direct to dataframe
from bq_helper import BigQueryHelper
# matplotlib for plotting
import matplotlib.pyplot as plt
import seaborn as sns
# use fivethirtyeight style for beautiful plot
plt.style.use('fivethirtyeight')


# # 1. Let's solve the Day 5 challenge
# > ** The challenge: ** How many commits (recorded in the "sample_commits" table) have been made in repos written in the Python programming language?

# In[ ]:


QUERY = """
    SELECT
        -- calculate the count of commit column in sample_commit table
        COUNT(sample_commit.commit) as count_of_commit_using_python
    FROM
      `bigquery-public-data.github_repos.sample_commits` AS sample_commit
      -- inner join with sample_files table with condition sample_commit.repo_name = sample_files.repo_name
    INNER JOIN `bigquery-public-data.github_repos.sample_files` AS sample_files
        ON sample_commit.repo_name = sample_files.repo_name
    WHERE
        -- where samples_files path column has .py extension
      sample_files.path LIKE '%.py'
        """

bq_assistant = BigQueryHelper("bigquery-public-data", "github_repos")
# I add a parametre called max_gb_scanned = 6 . This query size is more than 1 GB
df_commit_using_python_count = bq_assistant.query_to_pandas_safe(QUERY, max_gb_scanned=6)


# In[ ]:


df_commit_using_python_count


# # Answer:  31695737 commits have been made using Python. 

# In[ ]:




