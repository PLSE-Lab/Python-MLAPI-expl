#!/usr/bin/env python
# coding: utf-8

# # BigQuery: Getting started with BigQuery data on Kaggle
# BigQuery (https://cloud.google.com/bigquery) is a fully managed analytics database provided by Google Cloud. To get started with BQ data on kaggle, all you need is `google-cloud` and some understanding of SQL. Learn more about Standard SQL here: https://cloud.google.com/bigquery/docs/reference/standard-sql/
# 
# General structure of BigQuery database is:
# ```
# Project > n(dataset) > n(table) > n(partitions)
# ```
# Mostly you can disregrad paritions but then are helpful in logically storing your data and controlling cost on queries.
# 
# 

# In[ ]:


# all we need is bigquery
from google.cloud import bigquery

# there's no need for passing any credentials since access is managed by kaggle
bq = bigquery.Client()


# In[ ]:


# just write your sql query
sql_query = """
SELECT
  DISTINCT(word)
FROM
  `bigquery-public-data.samples.shakespeare`
WHERE
  word LIKE '%ing' OR word LIKE '%ed';
"""
# and run the query to get the result in a dataframe
results = bq.query(sql_query).to_dataframe()


# In[ ]:


# now this result dataframe can be used as you want
results.head(10)


# In[ ]:


# let's try another sample query
all_python_repos_activity_query = """
SELECT
  *
FROM
  `bigquery-public-data.samples.github_timeline`
WHERE
  repository_url LIKE '%python%'
"""
all_python_repos_activity_df = bq.query(all_python_repos_activity_query).to_dataframe()


# In[ ]:


# once you get the response dataframe, you can keep utilizing it as you would normally use pandas dataframe
all_python_repos_activity_df.head(10)


# In[ ]:


# let's do a normal describe
all_python_repos_activity_df.describe()


# In[ ]:


# let's get unique values in a column
all_python_repos_activity_df['repository_url'].unique()


# In[ ]:


# get smaller dataframe which contains one row per repository URL
unique_repositories_df = all_python_repos_activity_df.drop_duplicates(subset='repository_url', keep='last')


# In[ ]:


unique_repositories_df['repository_watchers'].plot()


# In[ ]:


unique_repositories_df.plot.scatter('repository_forks', 'repository_open_issues')


# In[ ]:


repo_languages_df = unique_repositories_df.groupby(['repository_language']).mean()
repo_languages_df.head(100)


# In[ ]:


repo_languages_df['repository_forks'].plot.bar()


# In[ ]:


repo_languages_df['repository_size'].plot.density()


# Querying to BQ is free on Kaggle but otherwise has an associated cost with it. It's always advised to copy your data to a dataframe rather than querying multiple times directy to the database. Beware of your memory limits though. Also, don't do `SELECT *`, always include the columns that you want in the dataframe.
# 
# Happy hacking!
