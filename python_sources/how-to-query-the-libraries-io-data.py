#!/usr/bin/env python
# coding: utf-8

# **How to Query the Libraries.io Data (BigQuery Dataset)**

# In[ ]:


import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
library = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="libraries_io")


# In[ ]:


bq_assistant = BigQueryHelper("bigquery-public-data", "libraries_io")
bq_assistant.list_tables()


# In[ ]:


bq_assistant.head("repositories", num_rows=20)


# In[ ]:


bq_assistant.table_schema("repositories")


# What are the repositories, avg project size, and avg # of stars?
# 
# 
# 

# In[ ]:


query1 = """
SELECT
  host_type,
  COUNT(*) repositories,
  ROUND(AVG(size),2) avg_size,
  ROUND(AVG(stars_count),2) avg_stars
FROM
  `bigquery-public-data.libraries_io.repositories`
GROUP BY
  host_type
ORDER BY
  repositories DESC
LIMIT
  1000;
        """
response1 = library.query_to_pandas_safe(query1)
response1.head(10)


# What are the top dependencies per platform?
# 
# 

# In[ ]:


query2 = """
SELECT
  dependency_platform,
  COUNT(*) dependencies,
  APPROX_TOP_COUNT(dependency_name, 3) top_dependencies
FROM
  `bigquery-public-data.libraries_io.dependencies`
GROUP BY
  dependency_platform
ORDER BY
  dependencies DESC;
        """
response2 = library.query_to_pandas_safe(query2, max_gb_scanned=10)
response2.head(20)


# What are the top unmaintained or deprecated projects?
# 
# 

# In[ ]:


query3 = """
SELECT
  name,
  repository_sourcerank,
  LANGUAGE,
  status
FROM
  `bigquery-public-data.libraries_io.projects_with_repository_fields`
WHERE
  status IN ('Deprecated',
    'Unmaintained')
ORDER BY
  repository_sourcerank DESC
LIMIT
  20;
        """
response3 = library.query_to_pandas_safe(query3, max_gb_scanned=10)
response3.head(20)


# Credit: Many functions are adapted from https://console.cloud.google.com/marketplace/details/libraries-io/librariesio

# In[ ]:




