#!/usr/bin/env python
# coding: utf-8

# # Analysis of Cooling Energy Impact on Elec.
# 
# - Clayton Miller
# - 29 Sept2018
# 
# First, we will load the `pandas` library

# In[ ]:


import pandas as pd


# In[ ]:


rawdata = pd.read_csv("../input/Office_Abigail.csv", parse_dates=True, index_col='timestamp')


# In[ ]:


rawdata.info()


# In[ ]:


rawdata.plot(figsize=(10,4))


# # Let's load the weather file!

# In[ ]:


weather_data = pd.read_csv("../input/weather0.csv", index_col='timestamp', parse_dates=True)


# In[ ]:


weather_data.head()


# In[ ]:


weather_data.info()


# In[ ]:


weather_hourly = weather_data.resample("H").mean()


# In[ ]:


weather_hourly.head()


# In[ ]:


weather_hourly["TemperatureC"].plot(figsize=(10,4))


# In[ ]:


weather_hourly_nooutlier = weather_hourly[weather_hourly > -40]


# In[ ]:


weather_hourly_nooutlier["TemperatureC"].plot(figsize=(10,4))


# # Great - we removed the outlier, now let's look at the info again

# In[ ]:


weather_hourly_nooutlier.info()


# In[ ]:


weather_hourly_nooutlier.head()


# # Let's fill the gap using `.fillna()`

# In[ ]:


weather_hourly_nooutlier_nogaps = weather_hourly_nooutlier.fillna(method='ffill')


# In[ ]:


weather_hourly_nooutlier_nogaps.info()


# # OK! Let's merge the weather (temperature) data with our electrical meter data

# In[ ]:


weather_hourly_nooutlier_nogaps['TemperatureC'].head()
rawdata = rawdata[~rawdata.index.duplicated(keep='first')]


# In[ ]:


rawdata['Office_Abigail'].head()


# In[ ]:


comparison = pd.concat([weather_hourly_nooutlier_nogaps['TemperatureC'],rawdata['Office_Abigail']], axis=1)


# In[ ]:


comparison.info()


# In[ ]:


comparison.plot(figsize=(20,10), subplots=True)


# # Let's compare the data using a scatter plot!

# In[ ]:


comparison.info()


# In[ ]:


comparison.plot(kind='scatter',x='TemperatureC', y='Office_Abigail', figsize=(10,10))


# In[ ]:


comparison.resample("D").mean().plot(kind='scatter',x='TemperatureC', y='Office_Abigail', figsize=(10,10))


# In[ ]:




