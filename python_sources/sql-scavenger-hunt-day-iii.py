#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import bq_helper
import pandas as pd

#object for the dataset
accidents=bq_helper.BigQueryHelper(active_project='bigquery-public-data',
                                   dataset_name="nhtsa_traffic_fatalities")


# In[ ]:


#lets see the list of the tables in the dataset
accidents.list_tables()


# In[ ]:


accidents.head("accident_2015")


# In[ ]:


accidents.head("vehicle_2016")


# Using Accident_2015 table..
# 
# **Which hours of the day do the most accidents occur during?**

# In[ ]:


accident_query = """SELECT COUNT(consecutive_number) as Count, 
                  EXTRACT(hour FROM timestamp_of_crash) as Hour_of_day
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY Hour_of_day
            ORDER BY COUNT(consecutive_number) DESC
        """
hourly_accident= accidents.query_to_pandas_safe(query)
hourly_accident


# I try plotting

# In[ ]:


import seaborn as sbn
sbn.barplot(x='Hour_of_day', y='Count', data=hourly_accident).set_title("Accidents by Hour")


# **Which state has the most hit and runs?**

# In[ ]:


hits_query = """SELECT registration_state_name, COUNT(hit_and_run)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run = "Yes"
            GROUP BY registration_state_name
            ORDER BY COUNT(hit_and_run) DESC
            """
hits= accidents.query_to_pandas_safe(hits_query)
hits


# In[ ]:




