#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# setup
import bq_helper

github = bq_helper.BigQueryHelper(active_project='bigquery-public-data',
                                 dataset_name='github_repos')
# explore tables
github.list_tables()


# **Question**
# 1. How many commits have been made in repos written in the Python programming language?

# In[ ]:


# total Python commits
query1 = '''SELECT COUNT(*) AS python_commits
        FROM `bigquery-public-data.github_repos.sample_files` AS f
        INNER JOIN `bigquery-public-data.github_repos.sample_commits` AS c ON c.repo_name = f.repo_name
        WHERE path LIKE '%.py'
        '''
github.query_to_pandas_safe(query1, max_gb_scanned=10)


# According to the dataset, there were 31,695,737 commits made in Python. I thought it'd also be interesting to see the breakdown of these commits based on `repo_name`.

# In[ ]:


# breakdown of python commits based on repo
query2 = '''SELECT c.repo_name, COUNT(*) AS commits
        FROM `bigquery-public-data.github_repos.sample_files` AS f
        INNER JOIN `bigquery-public-data.github_repos.sample_commits` AS c ON c.repo_name = f.repo_name
        WHERE path LIKE '%.py'
        GROUP BY 1
        ORDER BY 2 DESC
        '''
result2= github.query_to_pandas_safe(query2, max_gb_scanned=10)
result2


# I would like to try visualizing the results of this breakdown, but didn't have much success with bar plots or pie charts since the difference between highest-lowest was so great. The lowest values were barely visible in the resulting graphs. Please feel free to suggest alternatives for visualizing this! 

# In[ ]:




