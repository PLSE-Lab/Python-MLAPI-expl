#!/usr/bin/env python
# coding: utf-8

# <table>
#     <tr>
#         <td>
#         <center>
#         <font size="+5">SQL Scavenger Hunt : Day 2.</font>
#         </center>
#         </td>
#     </tr>
# </table>

# In[ ]:


# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")
hacker_news.head("full")


# In[ ]:


# query to pass to 
query = """SELECT parent, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY parent
            HAVING COUNT(id) > 10
        """


# In[ ]:


# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
popular_stories = hacker_news.query_to_pandas_safe(query)


# In[ ]:


popular_stories.head()


# # Scavenger hunt Questions:
# 
# * How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?
# * How many comments have been deleted? (If a comment was deleted the "deleted" column in the comments table will have the value "True".)
# * **Optional extra credit**: read about [aggregate functions other than COUNT()](https://cloud.google.com/bigquery/docs/reference/standard-sql/functions-and-operators#aggregate-functions) and modify one of the queries you wrote above to use a different aggregate function.
# 

# In[ ]:


# query to pass to 
full_query = """SELECT type, COUNT(id) as Stories
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type            
        """


# In[ ]:


# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
full_stories = hacker_news.query_to_pandas_safe(full_query)


# In[ ]:


full_stories


# In[ ]:


# query to pass to 
cd_query = """SELECT deleted , COUNT(*) as deleted_count
            FROM `bigquery-public-data.hacker_news.comments`            
            GROUP BY deleted  
            HAVING deleted = True
        """


# In[ ]:


# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
cd_stories = hacker_news.query_to_pandas_safe(cd_query)


# In[ ]:


cd_stories

