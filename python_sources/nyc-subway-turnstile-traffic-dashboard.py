#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import the libraries we need

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime, date, timedelta

import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot

import folium

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


init_notebook_mode()


# In[ ]:


# filename: data set to load from kernel
data_path = "../input/nys-turnstile-usage-data/turnstile-usage-data-2018.csv"

data = pd.read_csv(data_path, parse_dates=['Date'])


# In[ ]:


def fix_col_names(df):
    #Fix columns with trailing spaces
    df.columns = df.columns.str.strip()
    #Fix Columns with spaces in column name.
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]


# In[ ]:


fix_col_names(data)


# In[ ]:


data.head()


# In[ ]:


#narrow columns to only the ones that we're interested in.
data = data.filter(['unit', 'station', 'date', 'entries', 'exits'], axis=1)
# data = data.set_index(['station','date'])
data.tail(n=20)


# In[ ]:


stations_path = '../input/subway-station-locations-nyc/geocoded.csv'
stations = pd.read_csv(stations_path, 
                       usecols=[0, 2, 5, 6], 
                       names=['unit', 'station', 'latitude', 'longitude'])

stations.head()
#stations.iloc[:, 0]


# In[ ]:


len(stations.unit.unique()), len(data.unit.unique())


# In[ ]:


df = pd.merge(data, stations, how='inner', on='unit')
df.head()


# In[ ]:


len(df.unit.unique())


# In[ ]:


station_day_sum = (df.groupby(by='station_x')
                   .apply(lambda x: x.resample(rule='1D', on='date')
                   .agg({'entries': 'sum', 'exits': 'sum', 'latitude': 'first', 'longitude': 'first'}))
                  )

station_day_sum.head()


# In[ ]:


# Function to calculate several metrics on the dataset
# Now that it's grouped by station, date
def calculate_flow_stats(row):
    d = row.entries - row.exits
    return(row.entries + row.exits, d, abs(d))

## row function returns tuple of values that need to be spread out to each new column
def apply_flow_stats(df):
    (df['flowtotal'], df['flowdelta'],df['flowabs']) = zip(*df.apply(calculate_flow_stats, axis=1))

apply_flow_stats(station_day_sum)
station_day_sum.head()


# In[ ]:


most_recent = station_day_sum.index.get_level_values('date').max()
most_recent_5 = [most_recent - timedelta(i) for i in range(5)]
most_recent_5


# In[ ]:


last_day = station_day_sum[station_day_sum.index.get_level_values('date') == most_recent]
last_day.head()


# In[ ]:


last_week = station_day_sum[station_day_sum.index.get_level_values('date').isin(most_recent_5)]
last_week.head(n=10)


# In[ ]:


last_day.isna().sum()


# In[ ]:


last_day.dropna(inplace=True)
last_day.isna().sum()


# In[ ]:


fmap = folium.Map(location=[40.738, -73.94],
                        zoom_start=12,
                        tiles="CartoDB dark_matter")

for row, data in last_day.iterrows():
    
    if data[5] > 0: color = "#E37222" # tangerine
    else: color="#0A8A9F" # teal
        
    popup_text = "{}<br> total entries: {}<br> total exits: {}<br> net traffic: {}"
    popup_text = popup_text.format(row[0],
                            int(data[0]),
                            int(data[1]),
                            int(data[5]))

    latitude, longitude = data[2], data[3]
    radius = data[4] / 10000000000

    folium.CircleMarker(location=(latitude, longitude),
                            radius=radius,
                            color=color,
                            popup=popup_text,
                            fill=True).add_to(fmap)

fmap


# **Credits**
# 1. https://github.com/vincentropy/python_cartography_tutorial/blob/master/part1_basic_folium_maps.ipynb
# 2. https://www.kaggle.com/osunderdog/nyc-subway-data

# In[ ]:




