#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Libraries and environments
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.ticker import PercentFormatter
import seaborn as sns
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import warnings
warnings.filterwarnings('ignore')

pd.options.display.max_colwidth = 150


# In[ ]:


data = pd.read_csv("/kaggle/input/us-accidents/US_Accidents_May19.csv")


# In[ ]:


data.head()


# In[ ]:


data.columns


# Lets check percentage of missing value in each columns

# In[ ]:


data.shape


# In[ ]:


for name in data.columns:
    print(name, sum(data[name].isna())*100/2243939)


# Interesting! 70% of location data is missing. This may mean 70% of GPS got damaged in the accient.

# In[ ]:


pd.to_datetime(data['Start_Time']).dt.year.unique()


# In[ ]:


data['acc_year'] = pd.to_datetime(data['Start_Time']).dt.year
data['acc_month'] = pd.to_datetime(data['Start_Time']).dt.month
data['acc_hr_day'] = pd.to_datetime(data['Start_Time']).dt.hour


# In[ ]:


data['new_date'] = pd.to_datetime(data['Start_Time']).dt.date


# In[ ]:


temp = data.groupby('new_date')['ID'].count().reset_index()
fig = go.Figure()
fig.add_trace(go.Scatter(x=temp['new_date'], y=temp['ID']))

fig.update_layout(title_text='Accidents trend over the year',xaxis_rangeslider_visible=True)
fig.show()


# The Count of accidents have incresed over the time. Upon zooming with the slider one can easily see that there is consistent ups and downs in the graph.
# Looking at these dates when the accidents count was less. It was found that these were weekends. 
# 
# **Accidents are less on weekends compared to weekdays!**

# In[ ]:


data['day_name'] = pd.to_datetime(data['Start_Time']).dt.day_name()


# In[ ]:


temp = data.groupby(['acc_year', 'day_name'])['ID'].count().reset_index(name = 'count')
fig =go.Figure(go.Sunburst(
    labels=temp['day_name'].values,
    parents=temp['acc_year'].values,
    values=temp['count'].values,
))
fig.update_layout(margin = dict(t=0, l=0, r=0, b=0))

fig.show()


# Lets see which county has most number of accidents?

# In[ ]:


data.groupby('County')['ID'].count().reset_index(name = 'count').sort_values(by = 'count', ascending = False)


# Extra ordinary high number for Los Angeles! What are you doing folks? I will find out!

# In[ ]:


temp = data.groupby('County')['ID', 'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)',
       'Visibility(mi)', 'Wind_Speed(mph)',
       'Precipitation(in)'].agg({'ID':'count', 'Temperature(F)':'mean', 'Wind_Chill(F)':'mean', 'Humidity(%)':'mean', 'Pressure(in)':'mean',
       'Visibility(mi)':'mean', 'Wind_Speed(mph)':'mean',
       'Precipitation(in)':'mean'}).reset_index().sort_values(by = 'ID', ascending = False)


# In[ ]:


temp[temp['ID']>20000].head(20)


# In[ ]:


temp[temp['ID']<1000].head(20)


# In[ ]:


data.columns


# In[ ]:


temp = data.groupby('County')['ID', 'Amenity', 'Bump', 'Crossing',
       'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station',
       'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop' ].agg({'ID':'count', 'Amenity':'mean', 'Bump':'mean', 'Crossing':'mean',
       'Give_Way':'mean', 'Junction':'mean', 'No_Exit':'mean', 'Railway':'mean', 'Roundabout':'mean', 'Station':'mean',
       'Stop':'mean', 'Traffic_Calming':'mean', 'Traffic_Signal':'mean', 'Turning_Loop':'mean'}).reset_index().sort_values(by = 'ID', ascending = False)


# In[ ]:


temp = temp.head(20)
temp


# In[ ]:


temp.index = temp.County
temp.drop(['County', 'Turning_Loop'], axis = 1, inplace = True)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
for name in temp.columns:
    mx = MinMaxScaler()
    temp[name] = mx.fit_transform(temp[name].values.reshape(-1,1))


# In[ ]:


temp


# In[ ]:


plt.figure(figsize=(15,7))
plt.pcolor(temp)
plt.yticks(np.arange(0.5, len(temp.index), 1), temp.index)
plt.xticks(np.arange(0.5, len(temp.columns), 1), temp.columns)
plt.show()


# Most of accidents in Los Angeles are at Junction (Though I am not convined. Need to dive deep)
# For Harris it's Traffic Signal and Give Way. For Orange it's Crossing. 
# 
# I will dive into Los Angels data only to figure out what's happening!

# In[ ]:


data_la = data[data['County']=="Los Angeles"]


# In[ ]:


data_la.shape


# In[ ]:


temp = data_la.groupby('new_date')['ID'].count().reset_index()
fig = go.Figure()
fig.add_trace(go.Scatter(x=temp['new_date'], y=temp['ID']))

fig.update_layout(title_text='Accidents trend over the year',xaxis_rangeslider_visible=True)
fig.show()


# In[ ]:


temp['ID'].mean()


# Accidents are consistent over the years in LA. On an Avergae 156 Accidents happens every day in LA.

# In[ ]:


data_la['City'].value_counts()


# In[ ]:


data_la.columns


# In[ ]:


data_la[['ID', 'Start_Time', 'Start_Lat', 'Start_Lng',]].to_csv('data_la.csv', index = False)


# I will download the data for LA and view it on map as we have the coordinates to identify the hot spots of accident.

# Below you can see the how the accidents have grown in the Eastern region of Los Angeles. There are three main areas of accident.
# People travelling in these regions can be made aware about accident prone zone to reduce the accident.
# 

# In[ ]:


from IPython.display import YouTubeVideo
YouTubeVideo('1DXyTWV7pes')


# The mean statistics of the climatic features are not useful. We need to create categories of these features and then perform further analysis. 
# 
# Aligning this data with demographics might give us a better pucture. May be the cities in this county are overcrowded.
