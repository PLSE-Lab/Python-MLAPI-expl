#!/usr/bin/env python
# coding: utf-8

# # Get Started
# 
# After forking this notebook, run the code in the following cell:

# In[ ]:


# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("full")
#hacker_news.list_tables()


# # Question
# Using the Hacker News dataset in BigQuery, answer the following questions:
# 
# #### 1) How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?

# In[ ]:


import bq_helper

hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

query = """ SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
stories = hacker_news.query_to_pandas_safe(query)
print(stories.head())


# #### 2) How many comments have been deleted? (If a comment was deleted the "deleted" column in the comments table will have the value "True".)

# In[ ]:


import bq_helper
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")
query = """ SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY deleted
            HAVING deleted = True
        """

deleted_comments = hacker_news.query_to_pandas_safe(query)
print(deleted_comments.head())


# #### 3) Modify one of the queries you wrote above to use a different aggregate function.
# You can read about aggregate functions other than COUNT() **[in these docs](https://cloud.google.com/bigquery/docs/reference/standard-sql/functions-and-operators#aggregate-functions)**

# In[ ]:


import bq_helper

hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

query = """ SELECT type, COUNTIF(deleted = True)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
stories = hacker_news.query_to_pandas_safe(query)
print(stories.head())


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
