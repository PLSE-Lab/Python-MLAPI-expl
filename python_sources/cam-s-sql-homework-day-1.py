#!/usr/bin/env python
# coding: utf-8

# # Question1 - Which countries use a unit other than ppm to measure any type of pollution? 

# In[ ]:


import bq_helper

data = bq_helper.BigQueryHelper(active_project='bigquery-public-data',
                                dataset_name='openaq')



# In[ ]:


data.list_tables()


# In[ ]:


data.head("global_air_quality")


# In[ ]:


query1 = """ SELECT DISTINCT country, unit
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE LOWER(unit) != 'ppm'"""


# In[ ]:


data.estimate_query_size(query1)


# In[ ]:


q1_data = data.query_to_pandas_safe(query1)


# In[ ]:


q1_data


# # Question 2 - Which pollutants have a value of exactly 0?

# In[ ]:


query2 = """SELECT location, city,  pollutant, value
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0"""


# In[ ]:


data.estimate_query_size(query2)


# In[ ]:


data2 = data.query_to_pandas_safe(query2)


# In[ ]:


data2


# Please feel free to ask any questions you have in this notebook or in the [Q&A forums](https://www.kaggle.com/questions-and-answers)! 
# 
# Also, if you want to share or get comments on your kernel, remember you need to make it public first! You can change the visibility of your kernel under the "Settings" tab, on the right half of your screen.
