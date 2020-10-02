#!/usr/bin/env python
# coding: utf-8

# # Day 5 SQL homework - joins
# ## How many commits have been made in repos written in the Python programming language?
# - commits in  the "sample_commits" table
# - number of commits per repo for all the repos written in Python.
# - JOIN the sample_files and sample_commits 
# 
# Hint: You can figure out which files are written in Python by filtering results from the "sample_files" table using WHERE path LIKE '%.py'. This will return results where the "path" column ends in the text ".py", which is one way to identify which files have Python code.

# In[ ]:


import pandas as pd
import bq_helper


# In[ ]:


github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                dataset_name="github_repos")


# In[ ]:


github.list_tables()


# In[ ]:


github.head('sample_files')


# In[ ]:


github.head('sample_commits')


# In[ ]:


q = """
SELECT count(sc.commit) AS commit_count , sf.repo_name AS repository 
FROM `bigquery-public-data.github_repos.sample_commits` AS sc
INNER JOIN `bigquery-public-data.github_repos.sample_files` AS sf
ON sc.repo_name = sf.repo_name
WHERE sf.path LIKE '%.py'
GROUP BY repository
ORDER BY commit_count DESC
"""


# In[ ]:


github.estimate_query_size(q)


# In[ ]:


python_rep = github.query_to_pandas_safe(q, max_gb_scanned=6)


# In[ ]:


python_rep.head()

