#!/usr/bin/env python
# coding: utf-8

# ## 1) Which countries use a unit other than ppm to measure any type of pollution?
# 

# In[ ]:


import bq_helper as bq


# In[ ]:


open_aq = bq.BigQueryHelper(active_project="bigquery-public-data",
                           dataset_name="openaq")


# In[ ]:


open_aq.list_tables()


# In[ ]:


open_aq.head("global_air_quality")


# In[ ]:


query = """SELECT DISTINCT(country),
                  unit
           FROM `bigquery-public-data.openaq.global_air_quality`
           WHERE LOWER(unit) != "ppm" 
           ORDER BY country ASC
"""


# In[ ]:


not_ppm_countries = open_aq.query_to_pandas(query)


# In[ ]:


not_ppm_countries.head()


# In[ ]:


len(not_ppm_countries.country)


# In[ ]:


print(not_ppm_countries.country.tolist())


# ## 2) Which pollutants have a value of exactly 0?

# In[ ]:


query2 = """SELECT DISTINCT(pollutant)
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
            ORDER BY pollutant ASC
"""


# In[ ]:


zero_pollutants = open_aq.query_to_pandas_safe(query2)


# The below pollutants will have at least one value of zero.

# In[ ]:


zero_pollutants.head(10)

