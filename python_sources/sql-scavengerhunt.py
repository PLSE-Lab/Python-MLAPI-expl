#!/usr/bin/env python
# coding: utf-8

# SQL Scavengerhunt on OpenAQ dataset, which has information on air quality around the world.**
# 
# Day 1 Challenge:
# 
# 1. Which countries use a unit other than ppm to measure any type of pollution? 
# 2. Which pollutants have a value of exactly 0?

# In[ ]:


# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all tables presnet in the dataset
open_aq.list_tables()


# In[ ]:


# Inspect the first few rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")


# In[ ]:


# Query1 : Which countries use a unit other than ppm to measure any type of pollution?

query1 = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE pollutant != 'ppm'
        """


# In[ ]:


# only run this query if it's less than 100 MB in order to avoid exceeding the limit
open_aq.query_to_pandas_safe(query1, max_gb_scanned=0.1)


# In[ ]:


no_ppm = open_aq.query_to_pandas_safe(query1)


# In[ ]:


#The countries which do not use ppm to measure pollutants
no_ppm.country.unique()


# In[ ]:


# Which countries use a unit other than ppm to measure any type of pollution along with latitude and longitude
query1_1 = """SELECT country, latitude, longitude
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE pollutant != 'ppm'
        """


# In[ ]:


countries_map = open_aq.query_to_pandas_safe(query1_1, max_gb_scanned=0.1)


# In[ ]:


countries_map.head()


# In[ ]:


# Query2 : Which pollutants have a value of exactly 0?
query2 = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0.00
        """


# In[ ]:


# only run this query if it's less than 100 MB
open_aq.query_to_pandas_safe(query2, max_gb_scanned=0.1)


# In[ ]:


# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
zero_value = open_aq.query_to_pandas_safe(query2)


# In[ ]:


# check how big this query will be
open_aq.estimate_query_size(query2)


# In[ ]:


zero_value.pollutant.value_counts().head()


# In[ ]:


#Pollutants having zero value
zero_value.pollutant.unique()

