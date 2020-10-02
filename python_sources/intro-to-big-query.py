#!/usr/bin/env python
# coding: utf-8

# # Hacker News Data Analysis

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# Any results you write to the current directory are saved as output.


# In[ ]:


# import our bq_helper package
import bq_helper 


# In[ ]:


# create a helper object for our bigquery dataset
hacker_news = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "hacker_news")


# In[ ]:


# print a list of all the tables in the hacker_news dataset
hacker_news.list_tables()


# In[ ]:


# print information on all the columns in the "full" table
# in the hacker_news dataset
hacker_news.table_schema("full")


# In[ ]:


# preview the first couple lines of the "full" table
hacker_news.head("full")


# In[ ]:


# preview the first ten entries in the by column of the full table
hacker_news.head("full", selected_columns="by", num_rows=10)


# In[ ]:


# this query looks in the full table in the hacker_news
# dataset, then gets the score column from every row where 
# the type column has "job" in it.
query = """SELECT score
            FROM `bigquery-public-data.hacker_news.full`
            WHERE type = "job" """

# check how big this query will be
hacker_news.estimate_query_size(query)


# In[ ]:


# only run this query if it's less than 100 MB
hacker_news.query_to_pandas_safe(query, max_gb_scanned=0.1)


# In[ ]:


# check out the scores of job postings (if the 
# query is smaller than 1 gig)
job_post_scores = hacker_news.query_to_pandas_safe(query)


# In[ ]:


job_post_scores.head()


# In[ ]:


# average score for job posts
job_post_scores.score.mean()


# In[ ]:


job_post_scores = job_post_scores.dropna()


# In[ ]:


# save our dataframe as a .csv 
job_post_scores.to_csv("job_post_scores.csv")


# In[ ]:


job_type_query = """SELECT type, COUNT(*) as count
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type """

# check how big this query will be
hacker_news.estimate_query_size(job_type_query)


# In[ ]:


job_type = hacker_news.query_to_pandas_safe(job_type_query)


# In[ ]:


job_type


# In[ ]:


job_type.plot(x='type', y='count', kind = 'bar');


# In[ ]:




