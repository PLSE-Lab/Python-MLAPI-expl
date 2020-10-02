#!/usr/bin/env python
# coding: utf-8

# # How to Query Disclosed Standard Essential Patents (dSEP) Data (BigQuery)
# [Click here](https://www.kaggle.com/mrisdal/safely-analyzing-github-projects-popular-licenses) for a detailed notebook demonstrating how to use the bq_helper module and best practises for interacting with BigQuery datasets.

# In[ ]:


# Start by importing the bq_helper module and calling on the specific active_project and dataset_name for the BigQuery dataset.
import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package

dsep = bq_helper.BigQueryHelper(active_project="patents-public-data",
                                   dataset_name="dsep")


# In[ ]:


# View table names under the uspto_ptab data table
bq_assistant = BigQueryHelper("patents-public-data", "dsep")
bq_assistant.list_tables()


# In[ ]:


# View the first three rows of the disclosures_13 data table
bq_assistant.head("disclosures_13", num_rows=3)


# In[ ]:


# View information on all columns in the disclosures_13 data table
bq_assistant.table_schema("disclosures_13")


# ## Example SQL Query
# What are the harmonized versus unharmonized names of some of these patent owners?

# In[ ]:


query1 = """
SELECT DISTINCT
  patent_owner_harmonized, patent_owner_unharmonized
FROM
  `patents-public-data.dsep.disclosures_13`
LIMIT
  20;
        """
response1 = dsep.query_to_pandas_safe(query1)
response1.head(20)


# ## Importance of Knowing Your Query Sizes
# 
# It is important to understand how much data is being scanned in each query due to the free 5TB per month quota. For example, if a query is formed that scans all of the data in a particular column, given how large BigQuery datasets are it wouldn't be too surprising if it burns through a large chunk of that monthly quota!
# 
# Fortunately, the bq_helper module gives us tools to very easily estimate the size of our queries before running a query. Start by drafting up a query using BigQuery's Standard SQL syntax. Next, call the estimate_query_size function which will return the size of the query in GB. That way you can get a sense of how much data is being scanned before actually running your query.

# In[ ]:


bq_assistant.estimate_query_size(query1)


# Interpretting this number, this means my query scanned about ~0.001 GB (or 1 MB) of data in order to return a table of 20 harmonized and unharmonized patent owner names from the dataset.
