#!/usr/bin/env python
# coding: utf-8

# ## 1) How many commits have been made in repos written in the Python programming language?

# In[ ]:


import bq_helper as bq


# In[ ]:


github_repos = bq.BigQueryHelper(active_project = "bigquery-public-data",
                                      dataset_name = "github_repos")


# In[ ]:


query = """ SELECT F.repo_name,
                   COUNT(C.commit) AS commits
            FROM `bigquery-public-data.github_repos.sample_files` AS F
            INNER JOIN `bigquery-public-data.github_repos.sample_commits` AS C
                ON F.repo_name = C.repo_name
            WHERE F.path LIKE '%.py%'
            GROUP BY F.repo_name
            ORDER BY commits DESC
"""


# In[ ]:


github_repos.estimate_query_size(query)


# In[ ]:


query_results = github_repos.query_to_pandas_safe(query, max_gb_scanned=6)


# In[ ]:


query_results.head(10)


# ### Those numbers seem very high especially for `torvalds/linux` lets just double check if everything adds up by checking how many python files are in each of these repos. 

# In[ ]:


query2 = """SELECT repo_name,
                   COUNT(path) py_files
            FROM `bigquery-public-data.github_repos.sample_files`
            WHERE path LIKE '%.py' 
                      AND
                  repo_name IN ('Microsoft/vscode','facebook/react','apple/swift',
                                'tensorflow/tensorflow','torvalds/linux')
            GROUP BY repo_name
            ORDER BY py_files DESC
"""


# In[ ]:


github_repos.estimate_query_size(query2)


# In[ ]:


query2_results = github_repos.query_to_pandas_safe(query2,max_gb_scanned=6)


# In[ ]:


query2_results.head(10)


# ### So it looks like we need to form a query that gathers only the distinct repos with python in them, lets check if this will be the right method.

# In[ ]:


query3 = """WITH py_repos AS (
            SELECT DISTINCT(repo_name)
            FROM `bigquery-public-data.github_repos.sample_files`
            WHERE path LIKE '%.py')
                SELECT C.repo_name,
                       COUNT(commit) AS commits
                FROM `bigquery-public-data.github_repos.sample_commits` AS C
                INNER JOIN py_repos
                    ON C.repo_name = py_repos.repo_name
                GROUP BY C.repo_name
                ORDER BY commits DESC
"""


# In[ ]:


github_repos.estimate_query_size(query3)


# In[ ]:


commits_per_repo = github_repos.query_to_pandas_safe(query3, max_gb_scanned=6)


# In[ ]:


commits_per_repo.head(10)


# ### The table above is the correct number of commits, it looks like our first query was getting the repos multiplied by how many python files were present in the repo at that time.

# In[ ]:


commits_per_repo.sort_values('commits').plot.barh('repo_name','commits')

