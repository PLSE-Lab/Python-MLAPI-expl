#!/usr/bin/env python
# coding: utf-8

# # SQL Scavenger Hunt Day 2 Question
# 
# * How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?
# * How many comments have been deleted? (If a comment was deleted the "deleted" column in the comments table will have the value "True".)
# * Optional extra credit: [read about aggregate functions other than COUNT()](https://cloud.google.com/bigquery/docs/reference/standard-sql/functions-and-operators#aggregate-functions) and modify one of the queries you wrote above to use a different aggregate function.
# 

# ### Setting up Environment

# In[ ]:


# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print all the tables in this dataset
hacker_news.list_tables()


# In[ ]:


# print the first couple rows of the "stories" dataset
hacker_news.head("full")


# In[ ]:


#Added titles for the columns Nbr_of_Stories and type
query1=""" SELECT count(id) as Nbr_of_Stories,type
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY 2
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
num_of_stories=hacker_news.query_to_pandas_safe(query1)

num_of_stories


# In[ ]:


query2= """ SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY deleted
        """


# ### How many comments have been deleted? (If a comment was deleted the "deleted" column in the comments table will have the value "True".)

# In[ ]:


comments_deleted = hacker_news.query_to_pandas_safe(query2)
comments_deleted


# ## Optional extra credit
# Calculating Maximum score

# In[ ]:


#Extra points. Calculating Maximum score

query3=""" SELECT avg(score) AS average_score,author
           FROM `bigquery-public-data.hacker_news.stories`
           WHERE descendants>4 and text IS NOT NULL
          GROUP BY author
       """
avg_score=hacker_news.query_to_pandas_safe(query3)
avg_score

