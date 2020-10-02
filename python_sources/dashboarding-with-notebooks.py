#!/usr/bin/env python
# coding: utf-8

# # Collisions (accidents) in NYC by location
# In this notebook we will create some visualizations using daily updated dataset about collisions in NYC by location and injury.
# 
# 
# We will do 3 visualizations:
# 
# * Accidents location and related info including number of injured, killed, and vehicles involved over last 3 days.
# * Accidents map for full history of data
# * Time plot of number of accidents aggregated by number of injured and killed over time.

# In[ ]:


# Default imports
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# Custom imports
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# Data reading, wrangling, and getting hands dirty on Dashboarding through Pyplot

df = pd.read_csv('../input/nypd-motor-vehicle-collisions.csv', parse_dates=[['DATE', 'TIME']])

df = df.drop(columns=['BOROUGH','ZIP CODE','LOCATION','ON STREET NAME','CROSS STREET NAME','OFF STREET NAME','UNIQUE KEY', 
                      'NUMBER OF PEDESTRIANS INJURED', 'NUMBER OF CYCLIST INJURED', 'NUMBER OF MOTORIST INJURED', 
                      'NUMBER OF PEDESTRIANS KILLED', 'NUMBER OF CYCLIST KILLED', 'NUMBER OF MOTORIST KILLED'])

cols_dict = {'DATE_TIME': 'date', 'LATITUDE':'lat', 'LONGITUDE':'long', 'NUMBER OF PERSONS INJURED': 'inj', 'NUMBER OF PERSONS KILLED': 'kill', 
             'CONTRIBUTING FACTOR VEHICLE 1': 'vehcon1', 'CONTRIBUTING FACTOR VEHICLE 2': 'vehcon2', 'CONTRIBUTING FACTOR VEHICLE 3': 'vehcon3',
             'CONTRIBUTING FACTOR VEHICLE 4': 'vehcon4', 'CONTRIBUTING FACTOR VEHICLE 5': 'vehcon5', 'VEHICLE TYPE CODE 1': 'vehtype1',
             'VEHICLE TYPE CODE 1': 'vehtype1',  'VEHICLE TYPE CODE 2': 'vehtype2',  'VEHICLE TYPE CODE 3': 'vehtype3',  
             'VEHICLE TYPE CODE 4': 'vehtype4', 'VEHICLE TYPE CODE 5': 'vehtype5'}

df = df.rename(columns=cols_dict)


# In[ ]:


tmp = df.groupby([df.date.dt.date])[['inj','kill']].agg(['sum'])
tmp.columns = tmp.columns.droplevel(1)
tmp.index = pd.DatetimeIndex(tmp.index)

dateYearAgo = pd.Timestamp.now() - pd.DateOffset(365)

plt.figure(figsize=(30,10))

sns.lineplot(x=tmp[dateYearAgo:].index, y=tmp[dateYearAgo:].inj, label='Daily plot')
tmp[dateYearAgo.date():]['inj'].rolling(window=7, min_periods=1).mean().plot(style='co', label='Average of 7 days')
tmp[(tmp.index>=dateYearAgo) & (tmp.index.dayofweek>=5)]['inj'].plot(style='kD', label='Weekend')
plt.title('Number of injured per day for the past 365 days')
plt.xlabel('Date')
plt.ylabel('Number of injured')
plt.legend()
plt.show()


# In[ ]:


# import plotly
import plotly as py
import plotly.graph_objs as go

# these two lines are what allow your code to show up in notebook!
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

dateYearAgo = pd.Timestamp.now() - pd.DateOffset(365)

tmp = df.groupby([df.date.dt.date])[['inj','kill']].agg(['sum'])
tmp.columns = tmp.columns.droplevel(1)
tmp.index = pd.DatetimeIndex(tmp.index)
tmp1 = tmp[dateYearAgo:]['inj']

# sepcify that we want a scatter plot with, with date on the x axis and meet on the y axis
data = [go.Scatter(x=tmp1.index, y=tmp1)]

# specify the layout of our figure
layout = dict(title='Number of injured people by day',
              xaxis=dict(title='Date'), yaxis=dict(title='Number of injured'))

# create and show our figure
fig = dict(data = data, layout = layout)
iplot(fig)


# In[ ]:


tmp = df.groupby([df.date.dt.date])['inj'].agg(['sum'])
tmp.index = pd.DatetimeIndex(tmp.index)
tmp = tmp.rolling(window=30, min_periods=1).mean()['sum']

year = pd.Timestamp.now().year

plt.figure(figsize=(30,10))
sns.lineplot(x=tmp[pd.to_datetime(str(year)+'0101'):].index.dayofyear, y=tmp[pd.to_datetime(str(year)+'0101'):], label=str(year), ax=plt.gca())
sns.lineplot(x=tmp[pd.to_datetime(str(year-1)+'0101'):pd.to_datetime(str(year)+'0101')].index.dayofyear, 
             y=tmp[pd.to_datetime(str(year-1)+'0101'):pd.to_datetime(str(year)+'0101')], label=str(year-1), ax=plt.gca())
sns.lineplot(x=tmp[pd.to_datetime(str(year-2)+'0101'):pd.to_datetime(str(year-1)+'0101')].index.dayofyear, 
             y=tmp[pd.to_datetime(str(year-2)+'0101'):pd.to_datetime(str(year-1)+'0101')], label=str(year-2), ax=plt.gca())
sns.lineplot(x=tmp[pd.to_datetime(str(year-3)+'0101'):pd.to_datetime(str(year-2)+'0101')].index.dayofyear, 
             y=tmp[pd.to_datetime(str(year-3)+'0101'):pd.to_datetime(str(year-2)+'0101')], label=str(year-3), ax=plt.gca())
sns.lineplot(x=tmp[pd.to_datetime(str(year-4)+'0101'):pd.to_datetime(str(year-3)+'0101')].index.dayofyear,
             y=tmp[pd.to_datetime(str(year-4)+'0101'):pd.to_datetime(str(year-3)+'0101')], label=str(year-4), ax=plt.gca())
plt.title('Number of injured on average of 30 days over the years {0}-{1}'.format(year-4, year))
plt.xlabel('Date')
plt.ylabel('Number of injured')
plt.legend()
plt.show()


# In[ ]:


win, minYear, maxYear = 15, pd.Timestamp.now().year-4, pd.Timestamp.now().year
tmp = df.groupby([df.date.dt.date])['inj'].agg(['sum'])
tmp.index = pd.DatetimeIndex(tmp.index)
tmp = tmp.rolling(window=30, min_periods=1).mean()['sum']

data = []

for y in range(minYear,maxYear+1):
    tmp1 = tmp[tmp.index.year==y]
    data.append(go.Scatter(x=tmp1.index.dayofyear, y=tmp1, name=('Year ' + str(y))))

layout = dict(title='Nubmer of injured people on average of {0} days over the years {1}-{2}'.format(win, minYear, maxYear), xaxis=dict(title='Day of year'))

fig=dict(data=data, layout=layout)
iplot(fig)


# In[ ]:


import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

date = df.date.sort_values()[0].date()-pd.DateOffset(3)
tmp = df[(df.date >= pd.to_datetime(date)) & (df.date < (pd.to_datetime(date)+pd.DateOffset(3)))]

mapbox_access_token = 'pk.eyJ1IjoiYmlkZHkiLCJhIjoiY2pxNWZ1bjZ6MjRjczRhbXNxeG5udzkyNSJ9.xX6QLOAcoBmXZdUdocAeuA'

text = tmp.date.dt.floor('h').dt.strftime('%Y-%m-%d at %I %p.') + ' Fatal: ' +         tmp.kill.astype(int).astype(str) + ' | Injured: ' + tmp.inj.astype(int).astype(str)

data = [
    go.Scattermapbox(
        lat=tmp.lat,
        lon=tmp.long,
        mode='markers',
        marker=dict(
            size=5+tmp.inj*2+tmp.kill*3,
            color=tmp.date.dt.hour,
            colorscale='Earth',
            colorbar=dict(
                title='Time hour of accident'
            )
        ),
        hoverinfo='text',
        text=text,
    )
]

layout = go.Layout(
    autosize=True,
    hovermode='closest',
    title = 'NYPD Motor Vehicle Collisions since {0}'.format(str(date.date())),
    width=1400,
    height=700,
    mapbox=dict(
        accesstoken=mapbox_access_token,
        bearing=0,
        center=dict(
            lat=40.71,
            lon=-74.00
        ),
        pitch=0,
        zoom=10
    ),
)

fig = dict(data=data, layout=layout)
iplot(fig, filename='Multiple Mapbox')

