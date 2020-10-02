#!/usr/bin/env python
# coding: utf-8

# # Scavenger hunt
# ___
# 
# Now it's your turn! Here's the questions I would like you to get the data to answer:
# 
# * Which countries use a unit other than ppm to measure any type of pollution? (Hint: to get rows where the value *isn't* something, use "!=")
# * Which pollutants have a value of exactly 0?
# 
# In order to answer these questions, you can fork this notebook by hitting the blue "Fork Notebook" at the very top of this page (you may have to scroll up).  "Forking" something is making a copy of it that you can edit on your own without changing the original.

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


# # Scavenger hunt
# ___
# 
# Now it's your turn! Here's the questions I would like you to get the data to answer:
# 
# * Which countries use a unit other than ppm to measure any type of pollution? (Hint: to get rows where the value *isn't* something, use "!=")
# * Which pollutants have a value of exactly 0?
# 
# In order to answer these questions, you can fork this notebook by hitting the blue "Fork Notebook" at the very top of this page (you may have to scroll up).  "Forking" something is making a copy of it that you can edit on your own without changing the original.

# In[ ]:


# Your code goes here :)
# Which countries use a unit other than ppm to measure any type of pollution? 
query_1 = """SELECT distinct country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """
non_ppm = open_aq.query_to_pandas_safe(query_1)
non_ppm


# In[ ]:


# Your code goes here :
# Which pollutants have a value of exactly 0? 
query_2 = """SELECT distinct pollutant, value
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """
zero_city = open_aq.query_to_pandas_safe(query_2)
zero_city


# Please feel free to ask any questions you have in this notebook or in the [Q&A forums](https://www.kaggle.com/questions-and-answers)! 
# 
# Also, if you want to share or get comments on your kernel, remember you need to make it public first! You can change the visibility of your kernel under the "Settings" tab, on the right half of your screen.
