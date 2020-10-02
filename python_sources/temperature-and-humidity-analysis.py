#!/usr/bin/env python
# coding: utf-8

# # Basic Analysis of Temperature and Humidity Time Series data 

# In[ ]:


'''
Author: Ritwik Biswas
Description: Analysis of temperature time_series data
'''

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os


# ### Read CSV File

# In[ ]:


df = pd.read_csv('../input/temp-humidity.csv',header=0)
df.columns = ['Time', 'Temperature', 'Humidity']
df.head()


# ### Visualize Raw Dataset

# In[ ]:


# Temperature and Humidy Time Series
df.plot(title="Temperature and Humidity vs Time",figsize=(18, 10))
plt.show()


# In[ ]:


# Temperature vs Humidity
df.plot(x=1,y=2,kind="scatter",title="Temperature vs Humidity",figsize=(18, 10)) 
plt.show()


# ### Explore Correlation and Moving Average

# In[ ]:


print("Correlation Score: " + str(df['Temperature'].corr(df['Humidity'])))


# In[ ]:


# Calculate Moving Average with window 10
df['temperature_ma'] = df['Temperature'].rolling(1000).mean()
df['humidity_ma'] = df['Humidity'].rolling(1000).mean()
df.tail()


# In[ ]:


df.plot(y=[3,4],kind="line",title="Temperature vs Humidity",figsize=(18, 10)) 
plt.show()


# In[ ]:


df.plot(x=3,y=4,kind="scatter",title="Temperature_MA vs Humidity_MA",figsize=(18, 10)) 
plt.show()


# In[ ]:


print("Moving Average Correlation Score: " + str(df['temperature_ma'].corr(df['humidity_ma'])))


# In[ ]:


df.temperature_ma.plot.hist(alpha=0.5, figsize=(18, 10),title="Freq Distribution of Temp and Humidity",bins=50, legend=True)
df.humidity_ma.plot.hist(alpha=0.5,figsize=(18, 10), bins=50, legend=True)
plt.show()

