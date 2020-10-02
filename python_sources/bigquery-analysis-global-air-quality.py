#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from google.cloud import bigquery
from bq_helper import BigQueryHelper


# In[ ]:


# sample query from:
# https://cloud.google.com/bigquery/public-data/openaq#which_10_locations_have_had_the_worst_air_quality_this_month_as_measured_by_high_pm10
QUERY = """
        SELECT location, city, country, value, timestamp
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE pollutant = "pm10" AND timestamp > "2017-04-01"
        ORDER BY value DESC
        LIMIT 10
        """


# In[ ]:


bq_assistant = BigQueryHelper('bigquery-public-data', 'openaq')
# storing the extracted sample data (10 rows) in a Pandas dataframe
df = bq_assistant.query_to_pandas(QUERY)
df.head(3)


# Let's check all the columns in the database.

# In[ ]:


bq_assistant.query_to_pandas("""
        SELECT *
        FROM `bigquery-public-data.openaq.global_air_quality`
        LIMIT 2
        """)


# So, the database has 11 columns. Now I want to see how many unique countries are there in the data.

# In[ ]:


bq_assistant.query_to_pandas("""
        SELECT DISTINCT(country)
        FROM `bigquery-public-data.openaq.global_air_quality`
        ORDER BY country
        """)


# We have 64 countries in the database. Now I want to see how many records are there for each of these countries.

# In[ ]:


bq_assistant.query_to_pandas("""
        SELECT country, COUNT(*) AS Frequency
        FROM `bigquery-public-data.openaq.global_air_quality`
        GROUP BY country
        ORDER BY Frequency DESC
        """)


# Distinct Pollutants

# In[ ]:


bq_assistant.query_to_pandas("""
        SELECT DISTINCT(pollutant)
        FROM `bigquery-public-data.openaq.global_air_quality`
        """)


# Let's display the year-wise mean values for all the pollutants given in the dataset.

# In[ ]:


bq_assistant.query_to_pandas("""
        SELECT pollutant,
        EXTRACT(year FROM timestamp) AS Year, 
        AVG(value) AS Value
        FROM `bigquery-public-data.openaq.global_air_quality`
        GROUP BY Year, pollutant
        ORDER BY Year DESC
        """)


# ## Work in progress...
