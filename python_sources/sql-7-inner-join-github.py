#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")


# Write the code to answer the question below

# # Question
# 
# #### 1)  How many commits (recorded in the "sample_commits" table) have been made in repos written in the Python programming language? (I'm looking for the number of commits per repo for all the repos written in Python.
# * You'll want to JOIN the sample_files and sample_commits questions to answer this.
# * **Hint:** You can figure out which files are written in Python by filtering results from the "sample_files" table using `WHERE path LIKE '%.py'`. This will return results where the "path" column ends in the text ".py", which is one way to identify which files have Python code.

# In[ ]:


github.head('sample_commits',1)


# In[ ]:


github.head('sample_files',1)


# **1. SELECT ID AND COMMIT**

# In[ ]:


query = ''' SELECT  sf.id AS ID,
                    COUNT(sc.commit) AS commit                   
            FROM `bigquery-public-data.github_repos.sample_files` as sf
            INNER JOIN `bigquery-public-data.github_repos.sample_commits` as sc
                        ON sf.repo_name = sc.repo_name
            GROUP BY ID
        '''


# In[ ]:


commit_repos = github.query_to_pandas_safe(query, max_gb_scanned=20)


# In[ ]:


commit_repos.head(10)


# **2. SELECT ID (WHERE PATH IS PYTHON CODE) AND COMMIT**

# In[ ]:


github.head('sample_files',1)


# In[ ]:


queryy = ''' SELECT repo_name, id
             FROM `bigquery-public-data.github_repos.sample_files`
             WHERE path LIKE '%.py'
         '''


# In[ ]:


id_py = github.query_to_pandas(queryy)


# In[ ]:


id_py.head()


# In[ ]:


query2 = ''' WITH path_py AS
             (
                 SELECT repo_name, id
                 FROM `bigquery-public-data.github_repos.sample_files`
                 WHERE path LIKE '%.py'
             )
             
             SELECT 
                     py.id as ID, 
                     COUNT(sc.commit) as COMMIT 
             FROM path_py as py
             INNER JOIN `bigquery-public-data.github_repos.sample_commits` as sc
                         ON py.repo_name = sc.repo_name
             GROUP BY ID
             ORDER BY COMMIT
         '''


# In[ ]:


path_py_commit_id = github.query_to_pandas(query2)


# In[ ]:


path_py_commit_id


# # Feedback
# Bring any questions or feedback to the [Learn Discussion Forum](kaggle.com/learn-forum).
# 
# # Congratulations
# By now, you know all the key components to use BigQuery and SQL effectively.
# 
# Want to go play with your new powers?  Kaggle has BigQuery datasets available [here](https://www.kaggle.com/datasets?filetype=bigQuery).
