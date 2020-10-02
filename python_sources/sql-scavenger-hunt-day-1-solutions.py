#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import bq_helper


# In[ ]:


openaq = bq_helper.BigQueryHelper(active_project = 'bigquery-public-data', dataset_name='openaq')


# In[ ]:


openaq.list_tables()


# In[ ]:


openaq.head('global_air_quality')


# Select all the values from the "city" column for the rows where the "country" column is "us" (for "United States").

# In[ ]:


us_cities = openaq.query_to_pandas_safe("""
SELECT DISTINCT city
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE LOWER(country) = 'us'
ORDER BY city
""")


# In[ ]:


us_cities


# Which countries use a unit other than ppm to measure any type of pollution? (Hint: to get rows where the value isn't something, use "!=")

# In[ ]:


not_ppm = openaq.query_to_pandas_safe("""
SELECT DISTINCT country, unit
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE LOWER(unit) != 'ppm'
ORDER BY country
""")


# In[ ]:


not_ppm


# Which pollutants have a value of exactly 0?

# In[ ]:


absent_pollutants = openaq.query_to_pandas_safe("""
SELECT DISTINCT pollutant
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE value = 0.0
ORDER BY pollutant
""")


# In[ ]:


absent_pollutants

