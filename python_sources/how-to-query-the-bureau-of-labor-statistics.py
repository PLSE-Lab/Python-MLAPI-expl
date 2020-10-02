#!/usr/bin/env python
# coding: utf-8

# **How to Query the Bureau of Labor Statistics Dataset (BigQuery)**

# In[ ]:


import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
BLS = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="bls")


# In[ ]:


bq_assistant = BigQueryHelper("bigquery-public-data", "bls")
bq_assistant.list_tables()


# In[ ]:


bq_assistant.head('cpi_u', num_rows=3)


# In[ ]:


bq_assistant.table_schema("cpi_u")


# What is the average annual inflation across all US Cities?

# In[ ]:


query1 = """SELECT *, ROUND((100*(value-prev_year)/value), 1) rate
FROM (
  SELECT
    year,
    LAG(value) OVER(ORDER BY year) prev_year,
    ROUND(value, 1) AS value,
    area_name
  FROM
    `bigquery-public-data.bls.cpi_u`
  WHERE
    period = "S03"
    AND item_code = "SA0"
    AND area_name = "U.S. city average"
)
ORDER BY year
        """
response1 = BLS.query_to_pandas_safe(query1)
response1.head(10)


# What was the monthly unemployment rate (U3) in 2016?

# In[ ]:


query2 = """SELECT
  year,
  date,
  period,
  value,
  series_title
FROM
  `bigquery-public-data.bls.unemployment_cps`
WHERE
  series_id = "LNS14000000"
  AND year = 2016
ORDER BY date
        """
response2 = BLS.query_to_pandas_safe(query2)
response2.head(10)


# What are the top 10 hourly-waged types of work in Pittsburgh, PA for 2016?

# In[ ]:


query3 = """SELECT
  year,
  period,
  value,
  series_title
FROM
  `bigquery-public-data.bls.wm`
WHERE
  series_title LIKE '%Pittsburgh, PA%'
  AND year = 2016
ORDER BY
  value DESC
LIMIT
  10
        """
response3 = BLS.query_to_pandas_safe(query3, max_gb_scanned=10)
response3.head(10)


# Credit: Many functions are adaptations of https://cloud.google.com/bigquery/public-data/bureau-of-labor-statistics

# In[ ]:




