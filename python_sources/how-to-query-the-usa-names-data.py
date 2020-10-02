#!/usr/bin/env python
# coding: utf-8

# **How to Query the USA Names Data (BigQuery Dataset)**

# In[ ]:


import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
usa = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="usa_names")


# In[ ]:


bq_assistant = BigQueryHelper("bigquery-public-data", "usa_names")
bq_assistant.list_tables()


# In[ ]:


bq_assistant.head("usa_1910_current", num_rows=15)


# In[ ]:


bq_assistant.table_schema("usa_1910_current")


# What are the most common names?
# 

# In[ ]:


query1 = """
  SELECT
  names_step_1.name AS names_step_1_name,
  names_step_1.gender AS names_step_1_gender,
  COALESCE(CAST(SUM(names_step_1.number) AS FLOAT64),0) AS namesstep1totalpopulat_1
FROM
  `bigquery-public-data.usa_names.usa_1910_2013` AS names_step_1
GROUP BY
  1,
  2
ORDER BY
  3 DESC
LIMIT
  500;
        """
response1 = usa.query_to_pandas_safe(query1)
response1.head(50)


# What are the most common female names?
# 

# In[ ]:


query2 = """
  SELECT
  names_step_1.name AS names_step_1_name,
  COALESCE(CAST(SUM(names_step_1.number) AS FLOAT64),0) AS namesstep1totalpopulat_1
FROM
  `bigquery-public-data.usa_names.usa_1910_2013` AS names_step_1
WHERE
  (names_step_1.gender = 'F')
GROUP BY
  1
ORDER BY
  2 DESC
LIMIT
  500;
        """
response2 = usa.query_to_pandas_safe(query2)
response2.head(50)


# What are the names of 50 people in Texas?

# In[ ]:


query3 = """  
    SELECT name FROM `bigquery-public-data.usa_names.usa_1910_2013` 
    WHERE state = "TX" 
    LIMIT 100;
        """
response3 = usa.query_to_pandas_safe(query3)
response3.head(50)


# Credit: Many functions are adaptations from https://cloud.google.com/bigquery/public-data/usa-names
