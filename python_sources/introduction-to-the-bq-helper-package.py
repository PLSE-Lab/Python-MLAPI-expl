#!/usr/bin/env python
# coding: utf-8

# This kernel will provide a brief overview of [the bq_helper package](https://github.com/SohierDane/BigQuery_Helper/blob/master/bq_helper.py), which simplifies common read-only tasks in BigQuery by dealing with object references and unpacking result objects into pandas dataframes.
# 
# It currently only works here on Kaggle as it does not have any handling for the BigQuery authorization functions that Kaggle handles behind the scenes.
# 
# Please note that the bq_helper API is **NOT** yet stable; it is a work in progress and I may introduce backwards incompatible changes.

# In[ ]:


import pandas as pd
# https://github.com/SohierDane/BigQuery_Helper
from bq_helper import BigQueryHelper


# bq_helper requires the creation of one BigQueryHelper object per dataset. Let's make one now. We'll need to pass it two arguments:
# - The name of the BigQuery project, which on Kaggle should always be `bigquery-public-data`
# - The name of the dataset, which can be found in the dataset description.

# In[ ]:


bq_assistant = BigQueryHelper("bigquery-public-data", "openaq")


# The first thing I like to do with a dataset is to list all of the tables. This is a very small dataset, so there's just one.

# In[ ]:


bq_assistant.list_tables()


# Basic EDA is the next obvious step. `head` is implemented using the efficient `list_rows` function, so it will never use much data. Essentially, BigQuery only ever scans the results you see, while comparable results with a `SELECT` query would need to scan the entire table. You can comfortably use `head` as many times as you want on tables of any size.

# In[ ]:


bq_assistant.head("global_air_quality", num_rows=3)


# It would be nice to get some more details about the columns, let's review the table schema.

# In[ ]:


bq_assistant.table_schema("global_air_quality")


# Now we're ready to write a simple query. We should check how much memory it will scan. It won't matter in this case since the table is only 2 MB, but it's a good habit to get into for when you start working on larger datasets like [Github](https://www.kaggle.com/github/github-repos).

# In[ ]:


QUERY = "SELECT location, timestamp, pollutant FROM `bigquery-public-data.openaq.global_air_quality`"


# In[ ]:


bq_assistant.estimate_query_size(QUERY)


# That's roughly 0.4 MB. Won't even make a dent in our resource allocation of 5 TB. 
# 
# Now we can move on to actually running the query. We'll get the results back as a pandas DataFrame.

# In[ ]:


df = bq_assistant.query_to_pandas(QUERY)


# In[ ]:


df.head(3)


# If we were concerned about accidentally running large queries, there's a safe version of the query runner that will cancel large requests. In this first case, the query is well below the default scan limit of 1 GB and gets executed. 

# In[ ]:


df = bq_assistant.query_to_pandas_safe(QUERY)


# If we reduce the allowed scan size to the (insanely low) level of 1 kilobyte, the query will never be run.

# In[ ]:


df = bq_assistant.query_to_pandas_safe(QUERY, max_gb_scanned=1/10**6)


# Once you're ready to take full control of BigQuery and move beyond bq_helper, I'd recommend taking a look at both the [BigQuery API documentation](https://googlecloudplatform.github.io/google-cloud-python/latest/bigquery/reference.html) and the [bq_helper source code](https://github.com/SohierDane/BigQuery_Helper/blob/master/bq_helper.py). The [source code](https://github.com/SohierDane/BigQuery_Helper/blob/master/bq_helper.py) will help you understand what sections of the [BigQuery documentation](https://googlecloudplatform.github.io/google-cloud-python/latest/bigquery/reference.html) to read first.

# In[ ]:




