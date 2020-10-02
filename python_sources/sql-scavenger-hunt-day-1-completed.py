#!/usr/bin/env python
# coding: utf-8

# <table>
#     <tr>
#         <td>
#         <center>
#         <font size="+5">Day-1. SQL Scavenger Hunt</font>
#         </center>
#         </td>
#     </tr>
# </table>

# In[ ]:


# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()


# I'm going to take a peek at the first couple of rows to help me see what sort of data is in this dataset.

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


# Now I've got a dataframe called us_cities, which I can use like I would any other dataframe:

# In[ ]:


# What five cities have the most measurements taken there?
us_cities.city.value_counts().head()


# # Questions in Scavenger hunt- Day -1 
# 
# * Which countries use a unit other than ppm to measure any type of pollution? (Hint: to get rows where the value *isn't* something, use "!=")
# * Which pollutants have a value of exactly 0?
# 

# In[ ]:


# query to select all the items from the "country" column where the
# Pollutant unit measurement is not done in PPM.
country_query = """SELECT DISTINCT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm' 
        """


# In[ ]:


# # the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
countries = open_aq.query_to_pandas_safe(country_query)


# In[ ]:


countries


# In[ ]:


# query to select all the items from the "pollutant" column where the
# value measurement is equal to zero. 
pollutant_query = """SELECT DISTINCT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0 
        """


# In[ ]:


# # the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
pollutants = open_aq.query_to_pandas_safe(pollutant_query)


# In[ ]:


pollutants


# The Variables Countries and Pollutants produce the desired output to the questions asked. 
