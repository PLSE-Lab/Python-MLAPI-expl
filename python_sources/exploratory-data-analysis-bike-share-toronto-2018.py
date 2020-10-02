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


# Let's first import our data visualization libraries!

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import chart_studio.plotly as py
import plotly.express as px
import plotly.graph_objects as go


# In[ ]:


from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


# Since I'm going to do exploratory data analysis on the 2018 data, I'll load the following datasets:

# In[ ]:


q1 = pd.read_csv('/kaggle/input/toronto-bikeshare-data/bikeshare2018/bikeshare2018/Bike Share Toronto Ridership_Q1 2018.csv')
q2 = pd.read_csv('/kaggle/input/toronto-bikeshare-data/bikeshare2018/bikeshare2018/Bike Share Toronto Ridership_Q2 2018.csv')
q3 = pd.read_csv('/kaggle/input/toronto-bikeshare-data/bikeshare2018/bikeshare2018/Bike Share Toronto Ridership_Q3 2018.csv')
q4 = pd.read_csv('/kaggle/input/toronto-bikeshare-data/bikeshare2018/bikeshare2018/Bike Share Toronto Ridership_Q4 2018.csv')


# It's time to concatenate and see the head of the DataFrame:

# In[ ]:


toronto = [q1, q2, q3, q4]
bike_share = pd.concat(toronto)
bike_share.head()


# Before moving forward with the EDA, I'm going to convert the objects into DateTime format:

# In[ ]:


bike_share['trip_start_time'] = pd.to_datetime(bike_share['trip_start_time'])
bike_share['trip_stop_time'] = pd.to_datetime(bike_share['trip_stop_time'])


# Now I'm going to extract the hour, month and day of the week from both columns:

# In[ ]:


bike_share['hour_start'] = bike_share['trip_start_time'].apply(lambda time: time.hour)
bike_share['month_start'] = bike_share['trip_start_time'].apply(lambda time: time.month)
bike_share['weekday_start'] = bike_share['trip_start_time'].apply(lambda time: time.dayofweek)


# In[ ]:


bike_share['hour_stop'] = bike_share['trip_stop_time'].apply(lambda time: time.hour)
bike_share['month_stop'] = bike_share['trip_stop_time'].apply(lambda time: time.month)
bike_share['weekday_stop'] = bike_share['trip_stop_time'].apply(lambda time: time.dayofweek)


# In[ ]:


mon = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
bike_share['month_start'] = bike_share['month_start'].map(mon)
bike_share['month_stop'] = bike_share['month_stop'].map(mon)


# In[ ]:


day = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
bike_share['weekday_start'] = bike_share['weekday_start'].map(day)
bike_share['weekday_stop'] = bike_share['weekday_stop'].map(day)


# Here's the head of the DataFrame:

# In[ ]:


bike_share.head()


# So how did the Bike Share Toronto membership look in 2018?

# In[ ]:


plt.figure(figsize=(10,5))
sns.set_style('darkgrid')
sns.countplot(x='user_type',data=bike_share,palette='viridis')
plt.title('Bike Share Toronto Membership 2018')


# 82% are annual members!

# Ridership per month:

# In[ ]:


plt.figure(figsize=(10,5))
sns.set_style('darkgrid')
ridership = sns.countplot(data=bike_share, x='month_start', hue='user_type', palette='coolwarm')
plt.title('Bike Share Toronto Ridership 2018')
plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)


# The amount of rides increased as the weather got warmer. Ridership peaked during the summer months.

# Ridership per day:

# In[ ]:


plt.figure(figsize=(10,5))
sns.set_style('darkgrid')
ridership = sns.countplot(data=bike_share, x='weekday_start', hue='user_type', palette='coolwarm')
plt.title('Bike Share Toronto Ridership 2018')
plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)


# Annual members rode mostly during the week; especially from Tuesday to Thursday. Casual members were riding mostly on weekends.

# Ridership per hour:

# In[ ]:


plt.figure(figsize=(10,5))
sns.set_style('darkgrid')
ridership = sns.countplot(data=bike_share, x='hour_start', hue='user_type', palette='coolwarm')
plt.title('Bike Share Toronto Ridership 2018')
plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)


# Annual members rode between 8am-9am and 4pm-6pm; while casual members were riding throughout the day (noon to the early evening hours).

# Now the top stations for annual members:

# In[ ]:


plt.figure(figsize=(10,5))
sns.set_style('darkgrid')
sns.countplot(y=bike_share[bike_share['user_type']=='Annual Member']['from_station_name'],data=bike_share, palette='coolwarm',order=bike_share[bike_share['user_type']=='Annual Member']['from_station_name'].value_counts().index[:5])
plt.title('Top 5 Departing Stations (Annual Members)')


# In[ ]:


plt.figure(figsize=(10,5))
sns.set_style('darkgrid')
sns.countplot(y=bike_share[bike_share['user_type']=='Annual Member']['to_station_name'],data=bike_share, palette='coolwarm',order=bike_share[bike_share['user_type']=='Annual Member']['to_station_name'].value_counts().index[:5])
plt.title('Top 5 Arriving Stations (Annual Members)')


# Now the top stations for casual members:

# In[ ]:


plt.figure(figsize=(10,5))
sns.set_style('darkgrid')
sns.countplot(y=bike_share[bike_share['user_type']=='Casual Member']['from_station_name'],data=bike_share, palette='viridis',order=bike_share[bike_share['user_type']=='Casual Member']['from_station_name'].value_counts().index[:5])
plt.title('Top 5 Departing Stations (Casual Members)')


# In[ ]:


plt.figure(figsize=(10,5))
sns.set_style('darkgrid')
sns.countplot(y=bike_share[bike_share['user_type']=='Casual Member']['to_station_name'],data=bike_share, palette='viridis',order=bike_share[bike_share['user_type']=='Casual Member']['to_station_name'].value_counts().index[:5])
plt.title('Top 5 Arriving Stations (Casual Members)')


# I'm going to create a heat map to see the daily activity.
# 
# Before that happens, I'll group the DataFrame by the hour and the day of the week:

# In[ ]:


daily_activity = bike_share.groupby(by=['weekday_start','hour_start']).count()['user_type'].unstack()
daily_activity.head()


# In[ ]:


plt.figure(figsize=(12,6))
sns.heatmap(daily_activity,cmap='coolwarm')


# Most of the activity occured on Wednesday, Tuesday and Thursday at 5. 

# In[ ]:




