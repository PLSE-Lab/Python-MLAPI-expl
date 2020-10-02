#!/usr/bin/env python
# coding: utf-8

# ### Import the necessary libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Load the datasets into Pandas dataframes

# In[ ]:


df15 = pd.read_csv('../input/TimeSeries_TotalSolarGen_and_Load_IT_2015.csv',delimiter=',')
df16 = pd.read_csv('../input/TimeSeries_TotalSolarGen_and_Load_IT_2016.csv',delimiter=',')


# ### Convert the timestamp to proper format

# In[ ]:


df15.utc_timestamp = pd.to_datetime(df15.utc_timestamp)
df15.head()


# In[ ]:


df16.utc_timestamp = pd.to_datetime(df16.utc_timestamp, utc=True)
df16.head()


# ### Cleaning the data
# - Create columns for month, date, and time.
# - Drop columns of 'IT_load_new' and 'utc_timestamp'.

# In[ ]:


df15['month'] = df15['utc_timestamp'].dt.month
df15['date'] = df15['utc_timestamp'].dt.date
df15['time'] = df15['utc_timestamp'].dt.time
df15 = df15.drop(['utc_timestamp'],axis=1)
df15 = df15.drop(['IT_load_new'],axis=1)
df15.info()


# In[ ]:


df16['month'] = df16['utc_timestamp'].dt.month
df16['date'] = df16['utc_timestamp'].dt.date
df16['time'] = df16['utc_timestamp'].dt.time
df16 = df16.drop(['utc_timestamp'],axis=1)
df16 = df16.drop(['IT_load_new'],axis=1)
df16.info()


# ## A Look At the Power Generation

# In[ ]:


total15 = df15['IT_solar_generation'].sum()
print(total15)
total16 = df16['IT_solar_generation'].sum()
print(total16)
plt.bar(2015,total15)
plt.bar(2016,total16)
plt.xlabel('Year')
plt.xticks(np.arange(2015,2017))
plt.ylabel('Total power generation')
plt.title('Annual power generation')


# - Solar power generation was higher in 2015 than 2016.

# In[ ]:


df16_total_generation = df16.groupby(['month'])['IT_solar_generation'].sum()
df15_total_generation = df15.groupby(['month'])['IT_solar_generation'].sum()
months_values_16 = df16_total_generation.values
months_values_16 = months_values_16.reshape(-1,1)
months_values_15 = df15_total_generation.values
months_values_15 = months_values_15.reshape(-1,1)
months = np.arange(1,13).reshape(-1,1)
plt.plot(months,months_values_15,label='2015',linewidth='3')
plt.plot(months,months_values_16,label='2016',linewidth='3')
plt.xlabel('Month')
plt.ylabel('Total solar power generation')
plt.title('Cumilative Solar Power Generation (Monthly)')
plt.legend(loc='best')


# In[ ]:


df16_total_generation = df16.groupby(['month'])['IT_solar_generation'].mean()
df15_total_generation = df15.groupby(['month'])['IT_solar_generation'].mean()
months_values_16 = df16_total_generation.values
months_values_16 = months_values_16.reshape(-1,1)
months_values_15 = df15_total_generation.values
months_values_15 = months_values_15.reshape(-1,1)
months = np.arange(1,13).reshape(-1,1)
plt.plot(months,months_values_15,label='2015',linewidth='3')
plt.plot(months,months_values_16,label='2016',linewidth='3')
plt.xlabel('Month')
plt.ylabel('Average solar power generation')
plt.title('Mean Solar Power Generation (Monthly)')
plt.legend(loc='best')


# - Solar power generation peaked during the months of June, July, and August. 
# - As expected, there was very little power generated during winters.

# In[ ]:


df16_copy = df16
df15_copy = df15
df15_copy = df15_copy.pivot(index='date',columns='time')
df16_copy = df16_copy.pivot(index='date',columns='time')


# In[ ]:


plt.plot(df15_copy.IT_solar_generation.max(),label='2015',linewidth='3')
plt.plot(df16_copy.IT_solar_generation.max(),label='2016',linewidth='3')
plt.legend(loc='best')
plt.ylabel('Power generation')
plt.title('Max power generated')


# - Higher quantities of power was generated in 2016.
# - As expected, most of the power was generated during the afternoon.

# In[ ]:


plt.plot(df15_copy.IT_solar_generation.mean(),label='2015',linewidth='3')
plt.plot(df16_copy.IT_solar_generation.mean(),label='2016',linewidth='3')
plt.legend(loc='best')
plt.ylabel('Power generation')
plt.title('Mean power generation')


# - The mean power generation is quite similar. There was marginally better power generation in 2015.
# - This could be the cause for lower overall power genearation in 2016 compared to 2015.

# 
