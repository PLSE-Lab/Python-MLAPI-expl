#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
# https://github.com/SohierDane/BigQuery_Helper
from bq_helper import BigQueryHelper


# In[ ]:


bq_assistant = BigQueryHelper("bigquery-public-data", "github_repos")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'bq_assistant.list_tables()')


# # Number of Commits

# In[ ]:


QUERY = """
        SELECT COUNT(1)
        FROM `bigquery-public-data.github_repos.commits`
        """


# In[ ]:


get_ipython().run_cell_magic('time', '', 'df = bq_assistant.query_to_pandas_safe(QUERY)')


# In[ ]:


print('Size of dataframe: {} Bytes'.format(int(df.memory_usage(index=True, deep=True).sum())))


# In[ ]:


df.head()


# Therefore the number of **commits** is 203,318,560 i.e. around **203 million**

# # License Counts

# In[ ]:


QUERY = """
        SELECT license, COUNT(1)
        FROM `bigquery-public-data.github_repos.licenses`
        GROUP BY license
        ORDER BY 2 DESC
        """


# In[ ]:


get_ipython().run_cell_magic('time', '', 'df = bq_assistant.query_to_pandas_safe(QUERY)')


# In[ ]:


df.head()


# # Query the information about Tables

# In[ ]:


# print information on all the columns in the "commits" table
# in the github repos dataset
bq_assistant.table_schema("commits")


# # Number of encodings

# In[ ]:


#standardSQL
QUERY = """
SELECT
   encoding,COUNT(1)
FROM
  `bigquery-public-data.github_repos.commits`
GROUP BY encoding
ORDER BY 2 DESC
"""


# In[ ]:


get_ipython().run_cell_magic('time', '', 'df = bq_assistant.query_to_pandas_safe(QUERY)')


# In[ ]:


df.head()


# In[ ]:




