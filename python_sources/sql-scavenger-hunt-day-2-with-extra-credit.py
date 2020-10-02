#!/usr/bin/env python
# coding: utf-8

# <table>
#     <tr>
#         <td>
#         <center>
#         <font size="+1">If you haven't used BigQuery datasets on Kaggle previously, check out the <a href = "https://www.kaggle.com/rtatman/sql-scavenger-hunt-handbook/">Scavenger Hunt Handbook</a> kernel to get started.</font>
#         </center>
#         </td>
#     </tr>
# </table>
# 
# ___ 
# 
# ## Previous days:
# 
# * [**Day 1:** SELECT, FROM & WHERE](https://www.kaggle.com/rtatman/sql-scavenger-hunt-day-1/)
# 
# ____
# 

# # Scavenger hunt
# ___
# 
# Now it's your turn! Here's the questions I would like you to get the data to answer:
# 
# * How many stories (use the "id" column) are there are each type in the full table?
# * How many comments have been deleted? (If a comment was deleted the "deleted" column in the comments table will have the value "True".)
# * **Optional extra credit**: read about [aggregate functions other than COUNT()](https://cloud.google.com/bigquery/docs/reference/standard-sql/functions-and-operators#aggregate-functions) and modify one of the queries you wrote above to use a different aggregate function.

# In[ ]:


# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(
    active_project="bigquery-public-data",
    dataset_name="hacker_news"
)


# * How many stories (use the "id" column) are there are each type in the full table?

# In[ ]:


hacker_news.head("full")


# In[ ]:


query1 = """
    SELECT type, COUNT(id) AS total_by_type
    FROM `bigquery-public-data.hacker_news.full`
    GROUP BY type
    ORDER BY total_by_type DESC
"""
result1 = hacker_news.query_to_pandas_safe(query1)
result1


# * How many comments have been deleted? (If a comment was deleted the "deleted" column in the comments table will have the value "True".)

# In[ ]:


query2 = """
    SELECT COUNT(id) AS DELETED_COMMENTS
    FROM `bigquery-public-data.hacker_news.comments`
    WHERE deleted = True
"""
result2 = hacker_news.query_to_pandas_safe(query2)
result2[['DELETED_COMMENTS']]


# * **Optional extra credit**: read about [aggregate functions other than COUNT()](https://cloud.google.com/bigquery/docs/reference/standard-sql/functions-and-operators#aggregate-functions) and modify one of the queries you wrote above to use a different aggregate function.

# In[ ]:


query2_mod1 = """
    SELECT SUM(CAST(deleted as INT64)) AS DELETED_COMMENTS
    FROM `bigquery-public-data.hacker_news.comments`
"""
result2_mod1 = hacker_news.query_to_pandas_safe(query2_mod1)
result2_mod1[['DELETED_COMMENTS']]

