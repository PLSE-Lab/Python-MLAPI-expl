#!/usr/bin/env python
# coding: utf-8

# # Scavenger hunt
# ___
# 
# Now it's your turn! Here's the questions I would like you to get the data to answer:
# 
# * How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?
# * How many comments have been deleted? (If a comment was deleted the "deleted" column in the comments table will have the value "True".)
# * **Optional extra credit**: read about [aggregate functions other than COUNT()](https://cloud.google.com/bigquery/docs/reference/standard-sql/functions-and-operators#aggregate-functions) and modify one of the queries you wrote above to use a different aggregate function.
# 
# In order to answer these questions, you can fork this notebook by hitting the blue "Fork Notebook" at the very top of this page (you may have to scroll up). "Forking" something is making a copy of it that you can edit on your own without changing the original.

# In[ ]:


# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

query_1 = """SELECT type, COUNT(id) as cnt
                FROM `bigquery-public-data.hacker_news.full`
                GROUP BY type
          """
stoires_count = hacker_news.query_to_pandas_safe(query_1)
stoires_count


#  How many comments have been deleted? 

# In[ ]:


query_2 = """SELECT deleted, COUNT(id) as cnt
                FROM `bigquery-public-data.hacker_news.comments`
                WHERE deleted = True
                GROUP BY deleted
          """
delected_comments = hacker_news.query_to_pandas_safe(query_2)
delected_comments


#  Optional extra credit: read about [aggregate functions other than COUNT()]
#  (https://cloud.google.com/bigquery/docs/reference/standard-sql/functions-and-operators#aggregate-functions) 
#  and modify one of the queries you wrote above to use a different aggregate function.
# 

# In[ ]:


# what's the total score for book in stories table of which the auther is dead?
query_3 = """SELECT dead, sum(score) as total
                FROM `bigquery-public-data.hacker_news.stories`
                WHERE dead = True
                GROUP BY dead
          """
dead_score = hacker_news.query_to_pandas_safe(query_3)
dead_score


# Please feel free to ask any questions you have in this notebook or in the [Q&A forums](https://www.kaggle.com/questions-and-answers)! 
# 
# Also, if you want to share or get comments on your kernel, remember you need to make it public first! You can change the visibility of your kernel under the "Settings" tab, on the right half of your screen.
