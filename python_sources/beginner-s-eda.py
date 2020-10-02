#!/usr/bin/env python
# coding: utf-8

# ## A beginner's EDA on ASHRAE- Great Energy Predictor III

# #### In this notebook I have tried to explore multiple features and extract possible insights

# <img src="https://upload.wikimedia.org/wikipedia/en/f/fa/American_Society_of_Heating%2C_Refrigerating_and_Air-Conditioning_Engineers_Logo.svg">

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
import seaborn as sns
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)


# In[ ]:


train = pd.read_csv('../input/ashrae-energy-prediction/train.csv')


# In[ ]:


train.head()


# In[ ]:


building = pd.read_csv('../input/ashrae-energy-prediction/building_metadata.csv')


# In[ ]:


building.head()


# In[ ]:


results = train.join(building, how='outer',lsuffix='_x', rsuffix='_y')


# In[ ]:


weather_train = pd.read_csv('../input/ashrae-energy-prediction/weather_train.csv')


# In[ ]:


data = results.join(weather_train, how='outer',lsuffix='_X', rsuffix='_Y')


# In[ ]:


data.head()


# In[ ]:


#Drop redundant columns
del data['building_id_y']
del data['timestamp_Y']
del data['site_id_Y']


# In[ ]:


data.head()


# In[ ]:


data.describe()


# In[ ]:


len(data['building_id_x'].unique())


# In[ ]:


#There are 1449 Unique Building


# In[ ]:


groups = data.groupby('primary_use')['primary_use'].count()
groups.plot.bar(color='xkcd:lightish blue',alpha=0.5)


# In[ ]:


from matplotlib.pyplot import figure
figure(figsize=(30, 10))
groups = data.groupby('year_built')['year_built'].count()
groups.plot.bar(color='xkcd:lightish blue',alpha=0.5)


# In[ ]:


from plotly import graph_objs as go

fig = go.Figure()
for name, group in data.groupby('year_built'):
    trace = go.Histogram()
    trace.name = name
    trace.x = group['year_built']
    fig.add_trace(trace)


# In[ ]:


fig


# ##### Plot for # of buildings built in a particular year

# In[ ]:


sns.set(rc={'figure.figsize':(11,8)})
sns.set(style="whitegrid")
fig, ax = plt.subplots(1,1,figsize=(14, 6))
ax.set(xlabel='Year Built', ylabel='# Of Buildings', title='Buildings built in each year')
data['year_built'].value_counts(dropna=False).sort_index().plot(ax=ax)
# data['year_built'].value_counts(dropna=False).sort_index().plot(ax=ax)
ax.legend(['Train', 'Test']);


# In[ ]:


data.head()


# ##### Boxplots for Meter reading vs X

# In[ ]:


fig, axes = plt.subplots(1, 1, figsize=(14, 6))
sns.boxplot(x='site_id_X', y='meter_reading', data=data, showfliers=False);


# In[ ]:


fig, axes = plt.subplots(1, 1, figsize=(50, 20))
sns.boxplot(x='year_built', y='meter_reading', data=data, showfliers=False);


# In[ ]:


fig, axes = plt.subplots(1, 1, figsize=(14, 6))
sns.boxplot(x='site_id_X', y='air_temperature', data=data, showfliers=False);


# In[ ]:


fig, axes = plt.subplots(1, 1, figsize=(14, 6))
sns.boxplot(x='site_id_X', y='cloud_coverage', data=data, showfliers=False);


# In[ ]:


fig, axes = plt.subplots(1, 1, figsize=(14, 6))
sns.boxplot(x='site_id_X', y='dew_temperature', data=data, showfliers=False);


# ###### Scatter plot B/w Air Temperature and Meter Reading

# In[ ]:


sns.scatterplot(x='air_temperature', y='meter_reading', hue='year_built',data=data)


# ##### Scatter plot B/w Cloud Coverage and Meter Reading

# In[ ]:


sns.scatterplot(x='cloud_coverage', y='meter_reading', hue='year_built',data=data)


# ##### As cloud coverage increases Meter readings come down!

# In[ ]:


sns.scatterplot(x='air_temperature', y='meter_reading', data=data)


# ##### Boxplots for Meter reading for Floorcounts

# In[ ]:


fig, axes = plt.subplots(1, 1, figsize=(14, 6))
sns.boxplot(x='floor_count', y='meter_reading', data=data, showfliers=False);


# #### This is the end of my first introductory EDA

# In[ ]:




