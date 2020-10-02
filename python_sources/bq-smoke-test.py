#!/usr/bin/env python
# coding: utf-8

# This basic Python kernel shows you how to query the `commits` table in the GitHub Repos BigQuery dataset and look at the first ten commit messages it returns.

# In[ ]:


from google.cloud import bigquery
client = bigquery.Client()
QUERY = "SELECT commit, author.name FROM `bigquery-public-data.github_repos.commits` LIMIT 10"
query_job = client.query(QUERY)
iterator = query_job.result(timeout=30)
for row in list(iterator):
    print(row[0], row[1])

