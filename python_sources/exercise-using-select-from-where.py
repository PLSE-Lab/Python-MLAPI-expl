#!/usr/bin/env python
# coding: utf-8

# # Get Started
# Fork this notebook by hitting the blue "Fork Notebook" button at the top of this page.  "Forking" makes a copy that you can edit on your own without changing the original.
# 
# After forking this notebook, run the code in the following cell.

# In[1]:


# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()


# Then write and run the code to answer the questions below.

# # Question
# 
# #### 1) Which countries use a unit other than ppm to measure any type of pollution? 
# (Hint: to get rows where the value *isn't* something, use "!=")

# In[2]:


query = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm' 
        """


# In[21]:


countries = open_aq.query_to_pandas_safe(query, max_gb_scanned=0.1)
countries_values =[]
for c in countries.country:
    if c not in countries_values: countries_values.append(c)


# In[22]:



print (countries_values)


# #### 2) Which pollutants have a value of exactly 0?

# In[24]:


query = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0 
        """
pollutants = open_aq.query_to_pandas_safe(query, max_gb_scanned=0.1)


# In[28]:


pollutants_values =[]
for p in pollutants.pollutant:
    if p not in pollutants_values: pollutants_values.append(p)


# In[29]:


print(pollutants_values)


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
