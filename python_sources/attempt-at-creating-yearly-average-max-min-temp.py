#!/usr/bin/env python
# coding: utf-8

# **Pull Data from Every Month in a Year and Give them an Average**
# 
# I need to Query the database for the max temps and min temps for each month for every year designated. 
# 
# First we create the BigQueryHelper object.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd
# https://github.com/SohierDane/BigQuery_Helper
from bq_helper import BigQueryHelper

bq_assistant = BigQueryHelper(active_project= "bigquery-public-data", dataset_name= "noaa_gsod")


# Now we need to build the Query to pull only what we need. year/mo/max/min/temp
# Take a quick look to see if we have the right names. 

# In[ ]:


bq_assistant.list_tables()


# What I want is in those tables. 
# I'm thinking of different ways to do joins to bring just the averages of max, min, and temp, into the an avg_max, avg_min, avg_temp item, situated by year (and maybe station, but if I go that route I'll have to also pull lat and long). 
# 
# **Attempt to Query Average for a year**
# Starting with 1929 to create the first pandas dataframe. I will then append the rest through a loop.

# In[ ]:


QUERY = """
        SELECT year, AVG(data.temp) AS avg_temp, AVG(data.max) AS avg_max, AVG(data.min) AS avg_min
        FROM `bigquery-public-data.noaa_gsod.gsod1929` AS data
        GROUP BY year
        """

bq_assistant.estimate_query_size(QUERY)


# Run safe Query for data. 
# Using a local dataset, I found the data provided to be accurate. 

# In[ ]:


df = bq_assistant.query_to_pandas_safe(QUERY)
df.head()


# Time for the looped query and to append each response into the original 1929 dataframe.

# In[ ]:


START_YEAR = 1930
END_YEAR = 2019

for year in range(START_YEAR, END_YEAR):
   QUERY = """
        SELECT year, AVG(data.temp) AS avg_temp, AVG(data.max) AS avg_max, AVG(data.min) AS avg_min
        FROM `bigquery-public-data.noaa_gsod.gsod{}` AS data
        GROUP BY year
        """.format(year)
   df_temp = bq_assistant.query_to_pandas_safe(QUERY)
   df = df.append(df_temp, ignore_index=True)
   print ("Added {}".format(year))
df.head()


# Export into a files. I included indexed and non-indexed versions, as I wasn't sure which would be easier to use with my chosen visualization language. 

# In[ ]:


df.to_csv("avg_temps_1929_2018.csv", index=False)
df.to_csv("avg_temps_1929_2018_indexed.csv")
df.to_json("avg_temps_1929_2018_indexed.json")


# 

# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
