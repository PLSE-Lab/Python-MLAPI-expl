#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from google.cloud import bigquery
from bq_helper import BigQueryHelper


# In[2]:


QUERY = """SELECT
  state_name,
  COUNT(consecutive_number) AS accidents,
  SUM(number_of_fatalities) AS fatalities,
  SUM(number_of_fatalities) / COUNT(consecutive_number) AS fatalities_per_accident
FROM
  `bigquery-public-data.nhtsa_traffic_fatalities.accident_{0}`
GROUP BY
  state_name
ORDER BY
  fatalities_per_accident DESC"""


# In[3]:


bq_assistant = BigQueryHelper("bigquery-public-data", "nhtsa_traffic_fatalities")


# Let's check the size of the query before we run anything.

# In[8]:


bq_assistant.estimate_query_size(QUERY.format(2015))


# That's less than one MB, we should be able to run as many variations of it as we want.

# In[4]:


df_2015 = bq_assistant.query_to_pandas(QUERY.format(2015))


# In[5]:


df_2015.head()


# In[6]:


df_2016 = bq_assistant.query_to_pandas(QUERY.format(2016))


# In[7]:


df_2016.head()


# Looks like the Midwest has the most fatalities per accident. My best guess is that families tend to carpool more since they're driving longer distances than the more densely populated parts of the country, but we'd have to dig deeper into the data to get an informed answer.

# In[ ]:




