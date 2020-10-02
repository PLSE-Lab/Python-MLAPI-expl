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


open_aq.head("global_air_quality")


# # Which countries use a unit other than ppm to measure any type of pollution? 
# First lets try to find all the countries which use a unit other than ppm to measure pollution.

# In[ ]:


# Your code goes here :)

query2 = """
        SELECT country
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE unit != "ppm"
        """

no_ppm_countries = open_aq.query_to_pandas_safe(query2)


# In[ ]:


no_ppm_countries.head()


# In[ ]:


countries = no_ppm_countries['country'].unique()
print(countries)
print("No. of countries which use a unit other than ppm to measure pollution: " + str(len(countries)))


# # Which pollutants have a value of exactly 0?
# Now lets try to find out pollutants which have the value of exact 0.
# We will repeate the same steps again, just make small changes in the query.

# In[ ]:


query3 = """
        SELECT pollutant
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE value = 0
        """

zero_pollutants = open_aq.query_to_pandas_safe(query3)


# In[ ]:


zero_pollutants.head()


# In[ ]:


zero = zero_pollutants['pollutant'].unique()
print(zero)
print("No. of pollutants with zero value: " + str(len(zero)))


# Finally, we will export the data we queried in a csv file.

# In[ ]:


no_ppm_countries.to_csv("otp.csv")
zero_pollutants.to_csv("zero.csv")

