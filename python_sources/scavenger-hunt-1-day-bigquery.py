#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import bq_helper


# In[ ]:


open_aq = bq_helper.BigQueryHelper(active_project = "bigquery-public-data",dataset_name = "openaq")


# In[ ]:


open_aq.list_tables()


# In[ ]:


open_aq.head("global_air_quality")


# In[ ]:


query = """ select city from `bigquery-public-data.openaq.global_air_quality` where country = 'US' """


# In[ ]:


open_aq.estimate_query_size(query)


# In[ ]:


us_cities = open_aq.query_to_pandas_safe(query)


# In[ ]:


us_cities.head()


# In[ ]:


us_cities.city.value_counts().head()


# In[ ]:


open_aq.table_schema("global_air_quality")


# In[ ]:


query1 = """ select country from `bigquery-public-data.openaq.global_air_quality` 
             where unit != 'ppm'   """
open_aq.estimate_query_size(query1)


# In[ ]:


result = open_aq.query_to_pandas_safe(query1)


# In[ ]:


result.head()
result.country.value_counts()


# >  Total number of countries using unit other than 'ppm' are 64. The countries are :-

# In[ ]:


query1 = """ select distinct country,unit from `bigquery-public-data.openaq.global_air_quality` 
             where unit != 'ppm' order by country"""
open_aq.query_to_pandas_safe(query1)


# In[ ]:


query2 = """ SELECT pollutant FROM `bigquery-public-data.openaq.global_air_quality` WHERE value=0 """


# In[ ]:


open_aq.estimate_query_size(query2)


# In[ ]:


result2 = open_aq.query_to_pandas_safe(query2)


# In[ ]:


result2.head()
result2.pollutant.value_counts()
query = """ select distinct pollutant,value
            from `bigquery-public-data.openaq.global_air_quality` where value = 0 order by pollutant """


# > **Only 6 pollutants with value = 0**

# In[ ]:


open_aq.query_to_pandas_safe(query)


# In[ ]:




