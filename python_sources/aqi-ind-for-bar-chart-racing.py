#!/usr/bin/env python
# coding: utf-8

# The resultant CSV file is used for bar chart racing.
# 
# Click [here](https://preview.flourish.studio/2224319/x-MHQRaouTcxemSGADUCP_vYqcpI6bp8QIQ36FewJEv6KvxeWqapGJv86Gi9Uden/) for visualizing this data using bar chart racing.
# 
# Steps:
# 1. Go to [Flourish](https://app.flourish.studio/visualisation/2451165/edit)
# 2. Click on upload button and upload the AQI-IND.csv file.

# In[ ]:


import pandas as pd


# In[ ]:


df = pd.read_csv('../input/air-quality-data-in-india/city_day.csv',parse_dates=['Date'])
df.head()


# In[ ]:


rdf = df[(df['Date'] > '2020-03-10')] 
rdf.head()


# In[ ]:


squashed = rdf[['City','Date','AQI']]


# In[ ]:


squashed.fillna(method='bfill',inplace=True)
squashed.info()


# In[ ]:


pivoted = squashed.pivot_table(index='City',columns='Date',values='AQI')


# In[ ]:


pivoted.columns = pivoted.columns.strftime('%b-%d')


# In[ ]:


pivoted.head()


# In[ ]:


pivoted.to_csv('AQI-IND.csv')


# In[ ]:


# grouped = squashed.groupby(['City','Date'])['AQI'].sum()
# gdf = grouped.to_frame()
# gdf

