#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # we will use matplotlib to draw plot 
import csv as csv
from numpy import *
get_ipython().system('pip install plotly')
import plotly.plotly as py
import plotly.graph_objs as go


# ## Read datframe
# 

# In[ ]:


df = pd.read_csv('Data/HeatWaveConsolidatedDataset.csv')


# In[ ]:


df.head()


# ### Check for NA in Country and Date columns

# In[ ]:


df['Date (YMD)'].isnull().values.any()


# In[ ]:


df['Country'].isnull().values.any()


# ### Find country with max heatwaves (max entries in dataframe)

# In[ ]:


df_now = df.loc[df['Date (YMD)'] > '1973/1/1' ]


# In[ ]:


fig, ax = plt.subplots()
df_now['Country'].value_counts().plot(ax=ax, kind='bar')


# In[ ]:


df_mex = df_now.loc[df['Country'] == 'Mexico']
len(df_mex)


# ### This dataset contains a maximum number of heatwaves recorded in Mexico

# In[ ]:


print("Number of heatwaves recorded in Mexico from ", min(df_mex['Date (YMD)']) , " to ", max(df_mex['Date (YMD)']), " : ", len(df_mex))


# In[ ]:


df_mex.columns.values


# ## Plotting  heatwaves on a graph

# In[ ]:


df_weather = pd.read_csv('Data/chihuahua_mexico.csv')
#df_mex.plot()
#plt.show()
#fig, ax = plt.subplots()
#df_mex['Date'].value_counts().plot(ax=a)
#df_mex.value_counts().plot(ax='Date')
data = [df_weather.mo, df_weather.temp]
fig = dict(data=data)
py.iplot(fig, filename='simple-connectgaps')


# In[ ]:


df_weather = pd.read_csv('Data/chihuahua_mexico.csv')
print (df_weather)
df_weather = df_weather.assign(Date=pd.to_datetime(df_weather['year', 'mo', 'da'])
df_weather['Date'] = df_weather['Date'].strftime("%Y-%m-%d")print (df_weather)


# In[ ]:


df_weather.plot(x='mo', y='temp')


# In[ ]:





# In[ ]:




