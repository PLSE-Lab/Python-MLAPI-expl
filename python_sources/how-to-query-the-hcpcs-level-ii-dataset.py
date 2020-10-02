#!/usr/bin/env python
# coding: utf-8

# **How to Query the Healthcare Common Procedure Coding System (HCPCS) Level II
#  Dataset (BigQuery)**

# In[ ]:


import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
HCPCS2 = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="cms_codes")


# In[ ]:


bq_assistant = BigQueryHelper("bigquery-public-data", "cms_codes")
bq_assistant.list_tables()


# In[ ]:


bq_assistant.head("hcpcs", num_rows=3)


# In[ ]:


bq_assistant.table_schema("hcpcs")


# What are the descriptions for a set of HCPCS level II codes?

# In[ ]:


query1 = """SELECT
  HCPC,
  SEQNUM,
  RECID,
  LONG_DESCRIPTION,
  SHORT_DESCRIPTION
FROM
  `bigquery-public-data.cms_codes.hcpcs`
WHERE
  HCPC IN ('G0202',
    'A0998',
    'A4465',
    'A4565',
    'S9441' )
        """
response1 = HCPCS2.query_to_pandas_safe(query1)
response1.head(10)


# Credit: Many functions are adapted from https://cloud.google.com/bigquery/public-data/hcpcs-level2

# In[ ]:




