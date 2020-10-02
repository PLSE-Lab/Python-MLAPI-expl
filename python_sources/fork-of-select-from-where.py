#!/usr/bin/env python
# coding: utf-8

# **SELECT, FROM & WHERE**

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
#Important: Note that the argument we pass to FROM is not in single or double quotation marks (' or "). 
#It is in backticks (`). If you use quotation marks instead of backticks, you'll get this error when you 
#try to run the query: Syntax error: Unexpected string literal
query = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """


# In[ ]:


#Now I can use this query to get information from our open_aq dataset. I'm using the 
#BigQueryHelper.query_to_pandas_safe() method here because it won't run a query if it's 
#larger than 1 gigabyte, which helps me avoid accidentally running a very large query. 

# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
w_country = open_aq.query_to_pandas_safe(query)


# In[ ]:


print(w_country)


# In[ ]:


query = """SELECT DISTINCT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """


# In[ ]:


z_pollutant = open_aq.query_to_pandas_safe(query)


# In[ ]:


print(z_pollutant)

