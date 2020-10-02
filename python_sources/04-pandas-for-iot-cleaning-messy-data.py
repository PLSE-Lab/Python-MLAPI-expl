#!/usr/bin/env python
# coding: utf-8

# # Cleaning Messy Data
# 
# - Clayton Miller
# - Sept., 2018
# 
# In this analysis, we will take a look at a few different weather files
# 
# # First, let's load the pandas library
# 

# In[ ]:


import pandas as pd


# In[ ]:


rawdata = pd.read_csv("../input/weather0.csv", index_col='timestamp', parse_dates=True)


# In[ ]:


rawdata.head()


# In[ ]:


rawdata.info()


# # Let's first `resample` the data to hourly

# In[ ]:


rawdata_hourly = rawdata.resample("H").mean()


# In[ ]:


rawdata_hourly.info()


# In[ ]:


rawdata_hourly.plot(figsize=(20,10))


# In[ ]:


rawdata_hourly.plot(figsize=(20,15), subplots=True)


# # Remove the really low outliers

# In[ ]:


rawdata_hourly_nooutliers = rawdata_hourly[rawdata_hourly > -40]


# In[ ]:


rawdata_hourly_nooutliers.plot(figsize=(20,15), subplots=True)


# # Loop through all of the files using a `for` loop in Python

# In[ ]:


weatherfilelist = ["weather0.csv","weather4.csv","weather8.csv","weather10.csv"]


# In[ ]:


temp_data = []


# In[ ]:


for weatherfilename in weatherfilelist:
    print("Getting data from: "+weatherfilename)
    
    rawdata = pd.read_csv("../input/"+weatherfilename, index_col='timestamp', parse_dates=True)
    rawdata_hourly = rawdata.resample("H").mean()
    rawdata_hourly_nooutliers = rawdata_hourly[rawdata_hourly > -40]
    
    temperature = rawdata_hourly_nooutliers["TemperatureC"]
    temperature.name = weatherfilename
    
    temp_data.append(temperature)


# In[ ]:


all_temp_data = pd.concat(temp_data, axis=1)


# In[ ]:


all_temp_data.head()


# In[ ]:


all_temp_data.info()


# In[ ]:


all_temp_data.plot(figsize=(20,10))


# # Now we show the distributions of temperature data using boxplots and histograms

# In[ ]:


all_temp_data.boxplot(vert=False)


# In[ ]:


all_temp_data.plot.hist(figsize=(20,7), bins=50, alpha=0.5)


# In[ ]:




