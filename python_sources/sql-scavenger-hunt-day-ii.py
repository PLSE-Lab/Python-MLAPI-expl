#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import package with helper functions 
import bq_helper
# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",  dataset_name="hacker_news")
# Let see the tables in this dataset
hacker_news.list_tables()


# How does  a few rows of the full table look like?

# In[ ]:


hacker_news.head('full')


# How about a few rows of the comments table?

# In[ ]:


hacker_news.head('comments')


# We look at the stories table

# In[ ]:


hacker_news.head('stories')


# **How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?**

# In[ ]:


query = """
SELECT type, COUNT(id)
FROM `bigquery-public-data.hacker_news.full`
GROUP BY type
"""
full_table_type = hacker_news.query_to_pandas_safe(query)
full_table_type.head()


# In[ ]:


query_deleted = """
SELECT deleted, COUNT(id)
FROM `bigquery-public-data.hacker_news.comments`
WHERE deleted=True
GROUP BY deleted
"""
deleted = hacker_news.query_to_pandas_safe(query_deleted)
deleted

