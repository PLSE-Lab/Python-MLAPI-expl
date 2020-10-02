#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import bq_helper


# In[ ]:


hacker_news = bq_helper.BigQueryHelper(active_project = "bigquery-public-data",dataset_name = "hacker_news")


# In[ ]:


hacker_news.list_tables()


# In[ ]:


hacker_news.table_schema("full")


# In[ ]:


hacker_news.head("full")


# In[ ]:


hacker_news.head("full",selected_columns="by",num_rows = 10)


# In[ ]:


query = """ SELECT score from `bigquery-public-data.hacker_news.full` where type = "job" """
hacker_news.estimate_query_size(query)


# In[ ]:


hacker_news.query_to_pandas_safe(query, max_gb_scanned=0.1)


# In[ ]:


list = hacker_news.query_to_pandas_safe(query)


# In[ ]:


list.score.mean()


# In[ ]:


list.to_csv("FirstBigQuery.csv")


# In[ ]:




