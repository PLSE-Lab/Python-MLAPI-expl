#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import bq_helper

hn_dat = bq_helper.BigQueryHelper(active_project='bigquery-public-data',
                                dataset_name='hacker_news')


# In[ ]:


hn_dat.list_tables()


# # Question 1: How many unique stories are there of each type in the full table?
# 
# - obviously we will want to look at the stories table to answer this.

# In[ ]:


hn_dat.head('stories')


# - We want the number of unique ids in the first column.
# - I will therefore group by id and then count the number of results

# In[ ]:


q1 = """
SELECT type, COUNT(id)
FROM `bigquery-public-data.hacker_news.full`
GROUP BY type
"""


# Check that the query is not going to burn through an absurd amount of my monthly data quota

# In[ ]:


hn_dat.estimate_query_size(q1)


# only a few megabytes so we are likely safe

# In[ ]:


hn_summary = hn_dat.query_to_pandas_safe(q1)


# In[ ]:


hn_summary


# # Question 2: How many comments have been deleted? 
# 
# - For this one we need to work with the comments dataset

# In[ ]:


hn_dat.head('comments')


# In[ ]:


#(If a comment was deleted the "deleted" column in the comments table will have the value "True".)
q2 = """
SELECT deleted, COUNT(id)
FROM `bigquery-public-data.hacker_news.comments`
GROUP BY deleted
"""


# In[ ]:


comment_summary = hn_dat.query_to_pandas_safe(q2)


# In[ ]:


comment_summary

