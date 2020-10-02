#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()


# In[ ]:


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


us_cities.city.value_counts().head()


# In[ ]:


#Which countries use a unit other than ppm to measure any type of pollution? 
#(Hint: to get rows where the value isn't something, use "!=")

queryCountries = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'"""

countries = open_aq.query_to_pandas_safe(queryCountries)

countries.head()


# In[ ]:


countries.to_csv("countries.csv")


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize = (20, 6))
sns.countplot(countries['country'])


# In[ ]:


countries['country'].unique()


# In[ ]:


queryPollutant = """ SELECT pollutant 
                     FROM `bigquery-public-data.openaq.global_air_quality`
                     WHERE value = 0.00"""

pollutants_with_zero_value = open_aq.query_to_pandas_safe(queryPollutant)

pollutants_with_zero_value.head()


# In[ ]:


pollutants_with_zero_value.to_csv("pollutants.csv")


# In[ ]:


plt.figure(figsize = (20, 6))
sns.countplot(pollutants_with_zero_value['pollutant'])


# In[ ]:


pollutants_with_zero_value['pollutant'].unique()

