#!/usr/bin/env python
# coding: utf-8

# This is my first python notebook with bigquery dataset.Here I have done a basic overview with the help of super cool and useful kernel by Rachel on how to query bigquery datasets and as a part of SQL Scavenger Hunt-Day 1 ,i have tried to answer the two questions posted by the kaggle team.Happy learning !!!

# # Import the libraries 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper #For processing big queries


# # Reading the data

# In[ ]:


openAQ=bq_helper.BigQueryHelper(active_project="bigquery-public-data",dataset_name="openaq")


# # Check out the structure of the dataset

# In[ ]:


openAQ.list_tables()


# From the output we understand that there is only one table in Open AQ - that is "global air quality" table.Let us dig deeper.Let us understand about the columns in the table.

# In[ ]:


openAQ.table_schema('global_air_quality')


# From the output ,we infer that there are 11 columns will all of them accepting NULL values.(The 3rd part of the schema output describes this).Let us look at the first few row items.

# In[ ]:


openAQ.head('global_air_quality')


# Now that we have got an idea of how the dataset is structured,we prepare to run some queries over the data and interpret the output.The first question posted in the SQL Scavenger Hunt is this - "Which countries use a unit other than ppm to measure any type of pollution? " . From a simple SQL query we can narrow down on  this data.The logic is that we need to extract only country column which has unit other than ppm .As explained by Rachel we use the triple quotation marks to indicate that all the queries typed belong to one single string.The query is written as follows 

# In[ ]:


query = """ SELECT country,unit FROM `bigquery-public-data.openaq.global_air_quality` where unit !='ppm' order by country"""


# In[ ]:


#Check for query size 
openAQ.estimate_query_size(query)


# In[ ]:


#Running the query 
country=openAQ.query_to_pandas_safe(query)


# In[ ]:


# Which country has most number of units != ppm 
country.country.value_counts()


# Question 2 - Which pollutants have a value of exactly 0?

# In[ ]:


query = """ SELECT pollutant,value FROM `bigquery-public-data.openaq.global_air_quality` where value=0 order by pollutant"""


# In[ ]:


openAQ.estimate_query_size(query)


# In[ ]:


pollutants=openAQ.query_to_pandas_safe(query)


# In[ ]:


# Output first 5 rows
pollutants.head()


# In[ ]:


pollutants.pollutant.value_counts()


# This brings us to the close of Day 1 .. Awaiting for more exciting things ...
# 
# **If you like my work,pls upvote and encourage**
