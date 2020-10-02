#!/usr/bin/env python
# coding: utf-8

# # 1.  **What Rachel did...**

# In[ ]:


#import package with helper function
import bq_helper

#create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project='bigquery-public-data',
                                 dataset_name='openaq')

#print all the tables in this dataset
open_aq.list_tables()


# In[ ]:


#print first couple of rows
open_aq.head('global_air_quality')


# In[ ]:


#query to select all the items from the "city" column where the "country" column is is"us"
query = """SELECT city
           FROM `bigquery-public-data.openaq.global_air_quality`
           WHERE country = 'US'
        """


# In[ ]:


#query to pandas will only execute if it is smaller than one gigabyte
us_cities = open_aq.query_to_pandas_safe(query)


# In[ ]:


# What five cities have the most measurements taken there?
us_cities.city.value_counts().head()


# # 2. **My solutions to Rachel's  questions**

# # 2.1.  Which countries use a unit other than ppm to measure any type of pollution? 

# In[ ]:


query1 = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
         """


# In[ ]:


no_ppm_country = open_aq.query_to_pandas_safe(query1)


# In[ ]:


no_ppm_country.country.value_counts()


# There are 64 different countries listed above that do not use ppm unit for any type of pollution. 
# In FR (France) there are 2638 such different cities, in ES there are 1876 etc..

# Another solution inspired by [Brian W](https://www.kaggle.com/briwill):

# In[ ]:


query2 = """SELECT DISTINCT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
            ORDER BY country
         """


# In[ ]:


no_ppm_country = open_aq.query_to_pandas_safe(query2)


# In[ ]:


no_ppm_country


# ## 2.2. Which pollutants have a value of exactly 0?

# In[ ]:


query3 = """SELECT pollutant
           FROM `bigquery-public-data.openaq.global_air_quality`
           WHERE value = 0
        """


# In[ ]:


val = open_aq.query_to_pandas_safe(query3)


# In[ ]:


val.pollutant.value_counts()


# There are 7 pollutants with value exactly 0 listed above.

# Let us check in which cities are values 0.

# In[ ]:


query4 = """SELECT DISTINCT pollutant, city
           FROM `bigquery-public-data.openaq.global_air_quality`
           WHERE value = 0
           ORDER BY city
        """


# In[ ]:


val_city = open_aq.query_to_pandas_safe(query4)


# In[ ]:


val_city


# Please feel free to post any comments.
