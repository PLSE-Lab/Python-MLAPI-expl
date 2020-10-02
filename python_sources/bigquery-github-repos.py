#!/usr/bin/env python
# coding: utf-8

# ## Environment Setup and Overview

# In[ ]:


# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")


# **How many files are covered by each license?**

# We will be using two tables in the *GitHub Repos* data: *sample_files* and *licenses*. Let's have a look at each one.

# In[ ]:


github.head("sample_files")


# In[ ]:


github.head("licenses")


# In[ ]:


query = """
        SELECT L.license, count(sf.path) as number_of_files
        FROM `bigquery-public-data.github_repos.sample_files` as sf
        INNER JOIN `bigquery-public-data.github_repos.licenses` as L
            ON L.repo_name = sf.repo_Name
        GROUP BY L.license
        ORDER BY number_of_files DESC
        """
file_count_by_license = github.query_to_pandas_safe(query, max_gb_scanned = 6)
file_count_by_license


# ## SQL Scavenger Hunt: Day 5 Task

# **How many commits (recorded in the "sample_commits" table) have been made in repos written in the Python programming language? (number of commits per repo for all the repos written in Python)**

# In[ ]:


# explore the sample_commits table
github.head("sample_commits")


# In[ ]:


# use a Common Table Expression (CTE) to create a temporary table of Python-written repos
# then, execute the intended query on it
query = """
        WITH pyRepos AS
            (
                SELECT path, repo_name
                FROM `bigquery-public-data.github_repos.sample_files`
                WHERE path LIKE '%.py'
            )
        SELECT pr.repo_name, count(sc.commit) as number_of_commits
        FROM pyRepos as pr
        INNER JOIN `bigquery-public-data.github_repos.sample_commits` as sc
            ON pr.repo_name = sc.repo_name
        GROUP BY pr.repo_name
        ORDER BY number_of_commits DESC
        """
py_commits = github.query_to_pandas_safe(query, max_gb_scanned = 6)
py_commits

