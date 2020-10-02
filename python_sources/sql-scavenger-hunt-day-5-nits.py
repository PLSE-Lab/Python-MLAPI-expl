#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import bq_helper
github = bq_helper.BigQueryHelper(active_project = "bigquery-public-data",dataset_name="github_repos")


# In[ ]:


query = """ select L.license,count(o.path) as number_of_files from 
            `bigquery-public-data.github_repos.sample_files` as o
            inner join `bigquery-public-data.github_repos.licenses` as L 
            on o.repo_name = L.repo_name
            group by license
            order by number_of_files DESC
            """
github.estimate_query_size(query)
ans = github.query_to_pandas_safe(query,max_gb_scanned=6)
ans.head()


# **> Checking if i still know how to use LIKE in SQL ........... xD :P **

# In[ ]:


query2 = """ select count(path) from `bigquery-public-data.github_repos.sample_files`
            where path LIKE '%.py'
            """
github.estimate_query_size(query2)
github.query_to_pandas_safe(query2,max_gb_scanned = 4)


# **> Below Query returns number of commits per repo written in python only. First column is the python repo path and second column specifies the total amount commits**

# In[ ]:


LastQuery = """ WITH gits_py as
                (
                    select distinct repo_name from `bigquery-public-data.github_repos.sample_files` as p
                    where p.path LIKE '%.py'
                )
                select o.repo_name , count(o.commit) as CountCom 
                from `bigquery-public-data.github_repos.sample_commits` as o
                join gits_py on o.repo_name = gits_py.repo_name
                group by o.repo_name
                order by CountCom DESC
             """
github.estimate_query_size(LastQuery)


# In[ ]:


res = github.query_to_pandas_safe(LastQuery,max_gb_scanned = 6)
print(res)


# In[ ]:




