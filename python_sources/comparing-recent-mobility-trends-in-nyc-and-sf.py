#!/usr/bin/env python
# coding: utf-8

# # Comparing recent mobility trends in New York City and San Francisco
# 
# Using data from Google's Community Mobility Reports: https://www.kaggle.com/bigquery/covid19-google-mobility
# 
# and code that was adapted from https://www.kaggle.com/annaepishova/starter-google-community-mobility-reports

# In[ ]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import plotly.express as px
from google.cloud import bigquery
client = bigquery.Client()
dataset = client.get_dataset('bigquery-public-data.covid19_google_mobility')
tables = list(client.list_tables(dataset))


# In[ ]:


print('New York City')
sql = '''
SELECT
  *
FROM
  `bigquery-public-data.covid19_google_mobility.mobility_report` 
WHERE
  country_region = "United States"
  AND sub_region_1 = "New York"
  AND sub_region_2 = "New York County"
  AND date BETWEEN "2020-01-10" AND "2020-07-21"
ORDER BY
  date
'''
query_job = client.query(sql)
df = query_job.to_dataframe()
fig = plt.figure();
df.plot(x='date', rot=45, y=['retail_and_recreation_percent_change_from_baseline',                                  
                             'grocery_and_pharmacy_percent_change_from_baseline',
                             'parks_percent_change_from_baseline',
                             'transit_stations_percent_change_from_baseline',
                             'workplaces_percent_change_from_baseline',
                             'residential_percent_change_from_baseline'])
plt.legend(bbox_to_anchor=(1, 0.5), loc='lower left')
plt.xlabel('Date')
plt.ylabel('Percent Change From Baseline')
plt.show()


# In[ ]:


print('San Francisco')
sql = '''
SELECT
  *
FROM
  `bigquery-public-data.covid19_google_mobility.mobility_report` 
WHERE
  country_region = "United States"
  AND sub_region_1 = "California"
  AND sub_region_2 = "San Francisco County"
  AND date BETWEEN "2020-01-10" AND "2020-07-21"
ORDER BY
  date
'''
query_job = client.query(sql)
df = query_job.to_dataframe()
fig = plt.figure();
df.plot(x='date', rot=45, y=['retail_and_recreation_percent_change_from_baseline',                                  
                             'grocery_and_pharmacy_percent_change_from_baseline',
                             'parks_percent_change_from_baseline',
                             'transit_stations_percent_change_from_baseline',
                             'workplaces_percent_change_from_baseline',
                             'residential_percent_change_from_baseline'])
plt.legend(bbox_to_anchor=(1, 0.5), loc='lower left')
plt.xlabel('Date')
plt.ylabel('Percent Change From Baseline')
plt.show()


# In[ ]:




