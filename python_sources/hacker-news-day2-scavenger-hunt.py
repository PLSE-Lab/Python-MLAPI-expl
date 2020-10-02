#!/usr/bin/env python
# coding: utf-8

# **import libraries and call the hacker_news dataset object using bigquery**

# In[ ]:


import pandas as pd
from bq_helper import BigQueryHelper
bq_hacker_news = BigQueryHelper(active_project="bigquery-public-data",dataset_name="hacker_news")


# Let's check the data tables in hacker_news dataset

# In[ ]:


bq_hacker_news.list_tables()


# In[ ]:


bq_hacker_news.head('comments',num_rows=10)


# In[ ]:


query = """select parent,count(id) as total_replies
            from `bigquery-public-data.hacker_news.comments`
            group by parent
            having count(id)>10
        """
bq_hacker_news.estimate_query_size(query)


# In[ ]:


popular_stories = bq_hacker_news.query_to_pandas_safe(query)


# In[ ]:


popular_stories.head()


# Note that I have changed the column name to "total_replies" by using "as" in my query. This will assign the resulting column to "total_replies"(make sure you use underscore between words of a column name you like to specify).

# In[ ]:


bq_hacker_news.head('full',num_rows=50)


# In[ ]:


query = """select type,count(id) as total_stories
        from `bigquery-public-data.hacker_news.full`
        group by type
        """
bq_hacker_news.estimate_query_size(query)


# In[ ]:


stories = bq_hacker_news.query_to_pandas(query)
stories.head()


# In[ ]:


bq_hacker_news.table_schema('comments')


# In[ ]:


query = """select count(id) as total_deleted_comments
            from `bigquery-public-data.hacker_news.comments`
            where deleted=TRUE

        """
bq_hacker_news.estimate_query_size(query)


# In[ ]:


deleted = bq_hacker_news.query_to_pandas(query)
deleted.head()


# In[ ]:




