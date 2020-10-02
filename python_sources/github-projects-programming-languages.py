#!/usr/bin/env python
# coding: utf-8

# In this Jupyter notebook, we will do some EDA (Exploratory Data Analysis) about Programming Languages by querying the `languages` table in the GitHub Repos BigQuery dataset.

# In[ ]:


from google.cloud import bigquery
import pandas as pd
# https://github.com/SohierDane/BigQuery_Helper
from bq_helper import BigQueryHelper


# In[ ]:


# Use  bq_helper to create a BigQueryHelper object
bq_assistant = BigQueryHelper("bigquery-public-data", "github_repos")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'bq_assistant.table_schema("languages")')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'bq_assistant.head("languages", num_rows=20)')


# In[ ]:




