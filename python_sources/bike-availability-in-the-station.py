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


df1 = pd.read_csv('/kaggle/input/toronto-bikeshare-data/bikeshare-ridership-2017/2017 Data/Bikeshare Ridership (2017 Q1).csv')
df2 = pd.read_csv('/kaggle/input/toronto-bikeshare-data/bikeshare-ridership-2017/2017 Data/Bikeshare Ridership (2017 Q2).csv')
df3 = pd.read_csv('/kaggle/input/toronto-bikeshare-data/bikeshare-ridership-2017/2017 Data/Bikeshare Ridership (2017 Q3).csv')
df4 = pd.read_csv('/kaggle/input/toronto-bikeshare-data/bikeshare-ridership-2017/2017 Data/Bikeshare Ridership (2017 Q4).csv')


# In[ ]:


df2017 = df1.copy()
df2017 = df2017.append([df2, df3, df4], sort=False)
del df1
del df2
del df3
del df4


# In[ ]:


df2017.head(10)


# In[ ]:


# prepare the dataset
df2017['time'] = pd.to_datetime(df2017.trip_start_time)
df2017.index = df2017.time.dt.date
df2017.index.name = 'index'
# cleaning the dataset
df2017 = df2017.dropna(subset=['from_station_name','to_station_name'])
df = df2017.copy()


# In[ ]:



station = list(set(list(df.from_station_name.values) + list(df.to_station_name.values)))
date = df.index.unique().values


# In[ ]:


route = df.copy()
route = route[['from_station_name', 'to_station_name']]


# Separate two-way traveller with one-way traveller for each station

# In[ ]:


twoway = route.copy()
twoway = twoway[twoway.from_station_name == twoway.to_station_name]
oneway = route.copy()
oneway = oneway[oneway.from_station_name != oneway.to_station_name]


# In[ ]:


twoway_map = twoway.groupby('from_station_name').count().sort_values(by='to_station_name', ascending=False)
print('10 Stations with the highest number of two-way traveller')
twoway_map[:10]


# Create a table regarding to the number of bikes leaving ('from_station') and entering ('to_station') each day

# In[ ]:


# mapping the number of outgoing bike from each station each day in 2017
outmap = pd.get_dummies(route.from_station_name).groupby('index').sum()
# mapping the number of incoming bike to each station each day in 2017
inmap = pd.get_dummies(route.to_station_name).groupby('index').sum()


# In[ ]:


outmap.head(5) # number of bikes leaves the station


# In[ ]:


inmap.head(5) # number of bikes entering the station


# calculate the number of bikes entering the station minus number of bikes leaving the station
# * if the result >= 0 then there are enough bike available in the station to be used next morning
# * if the result < 0 then we need crew to return some bikes back to the station from other station

# In[ ]:


print('number of station with enough bike to use next morning, aka number of bikes entering > number of bikes leaving the station')
((inmap - outmap)>=0).sum(axis=1)


# In[ ]:


print('number of station with less bike to use next morning, or need a crew to return bikes back to station before next morning')
((inmap - outmap)<0).sum(axis=1)


# Total unique days in 2017 are 329 days. But as we can see below, some stations always lack of bikes for more than 200 days out of 329 days in a year.
# 
# May be we should add more bikes in the stations or require user/member to bring their own bikes

# In[ ]:


print('Station and the total number of days in 2017 where stations need more bikes to be returned by the crew every night')
((inmap - outmap)<0).sum(axis=0).sort_values(ascending=False)[:20]


# In[ ]:


bike_minus = inmap - outmap # incoming bikes minus leaving bikes
bike_minus = np.absolute(bike_minus[bike_minus < 0]) # show only minus value


# In[ ]:


bike_minus.head(10) # number of bikes that required by crew to be returned to each station


# In[ ]:


print('20 Stations with the highest number of required returned bikes in a day')
np.max(bike_minus, axis=0).sort_values(ascending=False)[:20]


# In[ ]:




