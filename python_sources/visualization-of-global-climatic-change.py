#!/usr/bin/env python
# coding: utf-8

# # 1. Visualizing the locations of all the stations

# **Input data**

# In[40]:


import numpy as np
import pandas as pd
from bq_helper import BigQueryHelper

bq_assistant = BigQueryHelper(active_project= "bigquery-public-data", dataset_name= "noaa_gsod")


# **Estimating how much memory the query need**

# In[41]:


QUERY = """
        SELECT lat, lon
        FROM `bigquery-public-data.noaa_gsod.stations`
        """
bq_assistant.estimate_query_size(QUERY)


# **Querying Data**

# In[42]:


df = bq_assistant.query_to_pandas_safe(QUERY)
df.head()


# **Projecting the stations into map
# **
# 
# [Detail of Basemap](https://matplotlib.org/basemap/api/basemap_api.html)

# In[91]:


from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

m = Basemap(projection='cyl',lat_0=0, lon_0=-100, resolution='l', area_thresh=1000.0)
m.drawmapboundary(fill_color='aqua')
m.fillcontinents(color='coral',lake_color='aqua')
parallels = np.arange(-90.,90,30.)
m.drawparallels(parallels,labels=[False,True,True,False])
meridians = np.arange(0.,350.,30.)
m.drawmeridians(meridians,labels=[True,False,False,True])
lon = df['lon'].tolist()
lat = df['lat'].tolist()
xpt,ypt = m(lon,lat)
m.plot(xpt,ypt,'b+') 
plt.gcf().set_size_inches(18.5, 10.5)
plt.show()


# # 2. Visualizing Global Averaged Temperature in 2017

# **Querying data**

# In[92]:


QUERY = """
        SELECT stn, avg_temp, lat,lon
        FROM(
            SELECT stn,AVG(data.temp) AS avg_temp
            FROM `bigquery-public-data.noaa_gsod.gsod2017` AS data
            GROUP BY stn
        )temp_
        INNER JOIN `bigquery-public-data.noaa_gsod.stations`AS stations
        ON temp_.stn = stations.usaf
        
        """
bq_assistant.estimate_query_size(QUERY)


# **Data cleaning**

# In[134]:


df = bq_assistant.query_to_pandas_safe(QUERY)
print(df.head())
df['lat'].isnull().value_counts()


# In[100]:


df = df.dropna(axis=0)
print('max_temp:%f'%df['avg_temp'].max())
print('min_temp:%f'%df['avg_temp'].min())


# **Visualization**
# 
# Here I only scatter the points with different colors that represent different temperature. If I have free time I will use interpolation to handle it.

# In[133]:


df_ = df.sample(5000)
m = Basemap(projection='cyl',lat_0=0, lon_0=-100, resolution='l', area_thresh=1000.0)
m.drawmapboundary()
m.drawcoastlines()
parallels = np.arange(-90.,90,30.)
m.drawparallels(parallels,labels=[False,True,True,False])
meridians = np.arange(0.,350.,30.)
m.drawmeridians(meridians,labels=[True,False,False,True])
lon = df_['lon'].tolist()
lat = df_['lat'].tolist()
df_['avg_temp'] = df_['avg_temp'].astype(int)
temp = df_['avg_temp'].tolist()
xpt,ypt = m(lon,lat)
cm = plt.cm.get_cmap('jet')
m.scatter(xpt, ypt, c=temp, cmap=cm, vmin = -70, vmax=100, s = 10, alpha = 0.4) 
plt.gcf().set_size_inches(18.5, 10.5)
plt.colorbar()
plt.show()


# **3. Plotting a dynamic picture of global temperature change**
# 
# Stay tuned
