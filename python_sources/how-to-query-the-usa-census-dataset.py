#!/usr/bin/env python
# coding: utf-8

# **How to Query the USA Census Dataset (BigQuery)**

# In[1]:


import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
census_data = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="census_bureau_usa")


# In[2]:


bq_assistant = BigQueryHelper("bigquery-public-data", "census_bureau_usa")
bq_assistant.list_tables()


# In[3]:


bq_assistant.head("population_by_zip_2010", num_rows=3)


# In[4]:


bq_assistant.table_schema("population_by_zip_2010")


# What are the ten most populous zip codes in the US in the 2010 census?

# In[5]:


query1 = """SELECT
  *
FROM
  `bigquery-public-data.census_bureau_usa.population_by_zip_2010`

        """
response1 = census_data.query_to_pandas_safe(query1)
response1.to_csv("population_by_zip_2010.csv")
response1.head(10)


# What are the top 10 zip codes that experienced the greatest change in population between the 2000 and 2010 censuses?

# In[ ]:


query2 = """SELECT
  *
FROM
  `bigquery-public-data.census_bureau_usa.population_by_zip_2000`

        """
response1 = census_data.query_to_pandas_safe(query2)
response1.to_csv("population_by_zip_2000.csv")
response1.head(10)


# ![https://cloud.google.com/bigquery/images/census-population-map.png](https://cloud.google.com/bigquery/images/census-population-map.png)
# https://cloud.google.com/bigquery/images/census-population-map.png

# In[ ]:




