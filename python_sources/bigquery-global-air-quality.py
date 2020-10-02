#!/usr/bin/env python
# coding: utf-8

# ## Overview

# In[ ]:


# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()


# In[ ]:


# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")


# In[ ]:


# query to select all the items from the "city" column where the
# "country" column is "us"
query = """SELECT city
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US'
        """


# In[ ]:


# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
us_cities = open_aq.query_to_pandas_safe(query)


# In[ ]:


# What five cities have the most measurements taken there?
us_cities.city.value_counts().head()


# ## SQL Scavanger Hunt: Day 1 Task

# **1. Which countries use a unit other than ppm to measure any type of pollution?**

# In[ ]:


# method 1
query = """SELECT country
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE unit != 'ppm'
        """
countries_NotPPM = open_aq.query_to_pandas_safe(query)
countries_NotPPM.country.value_counts()


# We have 64 countries that use units other than **ppm** to measure pollution.

# In[ ]:


# method 2
query = """
        SELECT DISTINCT country, unit
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE unit != 'ppm'
        ORDER BY country
        """

countries_NotPPM = open_aq.query_to_pandas_safe(query)
countries_NotPPM


# We have 64 countries that use units other than **ppm** to measure pollution.

# **2. Which pollutants have a value of exactly 0?**

# In[ ]:


# method 1
query = """
        SELECT pollutant
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE value = 0
        """

zero_pollutants = open_aq.query_to_pandas_safe(query)
zero_pollutants.pollutant.value_counts()


# We have 7 pollutants (shown above) that have exactly-zero values.

# In[ ]:


# method 2
query = """
        SELECT DISTINCT pollutant, value
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE value = 0
        ORDER BY pollutant
        """
zero_pollutants = open_aq.query_to_pandas_safe(query)
zero_pollutants


# We have 7 pollutants (shown above) that have exactly-zero values.
