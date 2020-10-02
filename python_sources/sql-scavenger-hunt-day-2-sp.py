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


# # Scavenger hunt
# ___
# 
# Now it's your turn! Here's the questions I would like you to get the data to answer:
# 
# * How many stories (use the "id" column) are there are each type in the full table?
# * How many comments have been deleted? (If a comment was deleted the "deleted" column in the comments table will have the value "True".)
# * **Optional extra credit**: read about [aggregate functions other than COUNT()](https://cloud.google.com/bigquery/docs/reference/standard-sql/functions-and-operators#aggregate-functions) and modify one of the queries you wrote above to use a different aggregate function.
# 

# In[ ]:


query1 = """
SELECT type,COUNT(id) AS type_count
FROM `bigquery-public-data.hacker_news.full`
GROUP BY type
ORDER BY type_count DESC
"""
query2 = """
SELECT COUNT(deleted) AS count_deleted
FROM `bigquery-public-data.hacker_news.comments`
"""

result1 = hacker_news.query_to_pandas_safe(query1)
result1


# In[ ]:


result2 = hacker_news.query_to_pandas_safe(query2)
result2 = result2.count_deleted.tolist()
print("deleted comments:",result2[0])

