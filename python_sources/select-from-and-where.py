#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()


# In[ ]:


open_aq.head('global_air_quality')


# Then write and run the code to answer the questions below.

# # Question
# 
# #### 1) Which countries use a unit other than ppm to measure any type of pollution? 
# (Hint: to get rows where the value *isn't* something, use "!=")

# In[ ]:


query = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE pollutant != 'ppm'
        """
countris_not_ppm = open_aq.query_to_pandas_safe(query)


# In[ ]:


# from now on countris_not_ppm is a dataframe which is nice :)
countris_not_ppm.head()


# In[ ]:


# countris_not_ppm.country.value_counts()
countris_not_ppm.country.unique()


# #### 2) Which pollutants have a value of exactly 0?

# In[ ]:


open_aq.head('global_air_quality')


# In[ ]:


query2 = """SELECT location
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0.0
        """
zero_pollutant = open_aq.query_to_pandas_safe(query2)


# In[ ]:


zero_pollutant.head(10)


# In[ ]:


# zero_pollutant.location.unique()


# # Keep Going
# After finishing this exercise, click [here](https://www.kaggle.com/dansbecker/group-by-having-count/).  You will learn about the **GROUP BY** command and its extensions.  This is especially valuable in large datasets like what you find in BigQuery.
# 
# # Help and Feedback 
# Bring any comments or questions to the [Learn Discussion Forum](https://www.kaggle.com/learn-forum).
# 
# If you want comments or help on your code, make it "public" first using the "Settings" tab on this page.
# 
# ---
# 
# *This tutorial is part of the [SQL Series](https://www.kaggle.com/learn/sql) on Kaggle Learn.*
