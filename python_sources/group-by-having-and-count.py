#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("full")


# # Question
# Using the Hacker News dataset in BigQuery, answer the following questions:
# 
# #### 1) How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?

# In[ ]:


query = """ SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """


# In[ ]:


group_type = hacker_news.query_to_pandas_safe(query)


# In[ ]:


group_type


# #### 2) How many comments have been deleted? (If a comment was deleted the "deleted" column in the comments table will have the value "True".)

# In[ ]:


hacker_news.head('comments')


# In[ ]:


query2 = """ SELECT deleted, COUNT(id)
             FROM `bigquery-public-data.hacker_news.comments`
             GROUP BY deleted
         """


# In[ ]:


deleted = hacker_news.query_to_pandas_safe(query2)


# In[ ]:


deleted


# In[ ]:


deleted_count = deleted.iloc[0,1]
deleted_count


# #### 3) Modify one of the queries you wrote above to use a different aggregate function.
# You can read about aggregate functions other than COUNT() **[in these docs](https://cloud.google.com/bigquery/docs/reference/standard-sql/functions-and-operators#aggregate-functions)**

# In[ ]:


query3 = """ SELECT deleted, MIN(id)
             FROM `bigquery-public-data.hacker_news.comments`
             GROUP BY deleted
         """


# In[ ]:


min_id = hacker_news.query_to_pandas_safe(query3)


# In[ ]:


min_id


# #### 4) How many comments have been deleted with parent > 1.000.000 ? (If a comment was deleted the "deleted" column in the comments table will have the value "True".)

# In[ ]:


query4 = """ SELECT deleted, COUNT(parent)
             FROM `bigquery-public-data.hacker_news.comments`
             GROUP BY deleted
             HAVING COUNT(parent)>1000000
         """  


# In[ ]:


deleted_1m = hacker_news.query_to_pandas_safe(query4)


# In[ ]:


deleted_1m # we do not have deleted comments that have parent > 1000000


# ---
# 
# # Keep Going
# [Click here](https://www.kaggle.com/dansbecker/order-by) to move on and learn about the ORDER BY clause.
# 
# # Feedback
# Bring any questions or feedback to the [Learn Discussion Forum](kaggle.com/learn-forum).
# 
# ----
# 
# *This exercise is part of the [SQL Series](https://www.kaggle.com/learn/sql) on Kaggle Learn.*
