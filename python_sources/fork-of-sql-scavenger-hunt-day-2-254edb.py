#!/usr/bin/env python
# coding: utf-8

# ### Scavenger hunt
# #### Question 1: How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?
# 
# 

# In[ ]:


# import package with helper functions 
import bq_helper

hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")
hacker_news.head('full')


# In[ ]:


# query to pass to 
query1 = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """


# In[ ]:


story_types = hacker_news.query_to_pandas_safe(query1)
story_types


# #### Question2:  How many comments have been deleted? (If a comment was deleted the "deleted" column in the comments table will have the value "True".)

# In[ ]:



query2 = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
        """
deleted_comments = hacker_news.query_to_pandas_safe(query2)
deleted_comments


#  **Optional extra credit**: read about [aggregate functions other than COUNT()](https://cloud.google.com/bigquery/docs/reference/standard-sql/functions-and-operators#aggregate-functions) and modify one of the queries you wrote above to use a different aggregate function.
# 

# In[ ]:




