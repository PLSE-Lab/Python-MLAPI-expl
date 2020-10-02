#!/usr/bin/env python
# coding: utf-8

# # Importing Packages and Datasets

# In[ ]:


# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "full" table
hacker_news.head("full")


# # Scavenger hunt 
# 
# ## Question 1 : How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?
# 

# In[ ]:


query_1 = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            ORDER BY COUNT(id) desc
        """
id_vs_type = hacker_news.query_to_pandas_safe(query_1)

id_vs_type.head(10)


# # Scavenger hunt 
# 
# ## Question 2 : ****How many comments have been deleted? (If a comment was deleted the "deleted" column in the comments table will have the value "True".)
# 

# In[ ]:


# Answer to Question 2 : How many comments have been deleted? (If a comment was deleted the "deleted" column in the comments table will have the value "True".)

query_2 = """SELECT COUNT(id)
             FROM `bigquery-public-data.hacker_news.full`
             GROUP BY type, deleted
             HAVING type = 'comment'
             AND deleted = True            
                        
        """


# In[ ]:


no_of_comments_deleted = hacker_news.query_to_pandas_safe(query_2)

no_of_comments_deleted.head(10)


# # Scavenger hunt 
# ## Optional extra credit: read about aggregate functions other than COUNT() and modify one of the queries you wrote above to use a different aggregate function.
# 

# In[ ]:


query2_modified = """
                        SELECT SUM(CAST(deleted as INT64)) 
                        FROM `bigquery-public-data.hacker_news.full`
                        WHERE type = 'comment'
"""
result2 = hacker_news.query_to_pandas_safe(query2_modified)
result2.head()


# In[ ]:




