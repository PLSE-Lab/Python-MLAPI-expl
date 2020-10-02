#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


files = ['../input/accidents_2005_to_2007.csv','../input/accidents_2009_to_2011.csv','../input/accidents_2012_to_2014.csv']
frame_list = []
for file in files:
    df = pd.read_csv(file, parse_dates=[9,11])
    frame_list.append(df)
accident_data = pd.concat(frame_list)
print(accident_data.size)


# In[ ]:


accident_data.dtypes


# In[ ]:


#Let Folium determine the scale
accident_loc_data = accident_data[accident_data['Speed_limit']==30] # Reducing data size so it runs faster
accident_loc_data = accident_loc_data[accident_loc_data['Year']==2007] # Reducing data size so it runs faster
accident_loc_data = accident_loc_data[accident_loc_data['Number_of_Vehicles']==4] # Reducing data size so it runs faster
accident_loc_data = accident_loc_data[['Latitude', 'Longitude']]
accident_loc_data = accident_loc_data.dropna(axis=0, subset=['Latitude','Longitude'])
accident_loc_data.describe()


# In[ ]:


pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
accident_data.head()


# In[ ]:


accident_data.Junction_Detail.unique()


# In[ ]:


import folium
locations = accident_loc_data[['Latitude', 'Longitude']]
locations['Latitude'] = locations['Latitude'].astype(float)
locations['Longitude'] = locations['Longitude'].astype(float)
locationlist = locations.values.tolist()
print(len(locationlist))
print(locationlist[17])
map = folium.Map(location=[51.5074, -0.1278], zoom_start=12)
for point in range(0, len(locationlist)):
   folium.Marker(locationlist[point]).add_to(map)
map


# In[ ]:


# Accidents by Speed Limit
accident_data.Speed_limit.value_counts().sort_index()


# In[ ]:


# Accidents by Speed Limit with causuality
#accident_data.Number_of_Casualties.value_counts().sort_index()
#accident_data[accident_data['Number_of_Casualties']>0].Speed_limit.value_counts().sort_index()


# In[ ]:


accident_data.Weather_Conditions.value_counts()


# In[ ]:


# If we analyze data by hour of the day, we see more accidents happening during office hours
accidents_by_time = accident_data.Time.value_counts()
accidents_by_hour = accidents_by_time.groupby(accidents_by_time.index.hour).sum()
plot = accidents_by_hour.plot('bar', figsize=(20,8))
plot.set_xlabel("Hour of Day")
plot.set_ylabel("No of Accidents")


# In[ ]:


# If we analyze data by day of the week
accidents_by_dow = accident_data.Day_of_Week.value_counts()
plot = accidents_by_dow.plot('bar', figsize=(20,8))
plot.set_xlabel("Day of Week")
plot.set_ylabel("No of Accidents")


# In[ ]:


# If we analyze data by hour of the day on Friday and Saturday, we do not see accidents during drinking hours so
# our assumption about more accidents during weekend evenings does not hold true
accidents_by_time = accident_data[accident_data['Day_of_Week'].isin([5,6])].Time.value_counts()
accidents_by_hour = accidents_by_time.groupby(accidents_by_time.index.hour).sum()
plot = accidents_by_hour.plot('bar', figsize=(20,8))
plot.set_xlabel("Hour of Day")
plot.set_ylabel("No of Accidents")


# In[ ]:


# Find 20 centers based on accident locations, these can be good points for emergency response stations
from sklearn.cluster import KMeans

acc_locations = accident_data.dropna(axis=0, subset=['Latitude','Longitude'])
kmeans = KMeans(n_clusters=20, random_state=0).fit(acc_locations[['Latitude', 'Longitude']])

centers = kmeans.cluster_centers_
#print(centers)
map = folium.Map(location=[53.5074, -0.1278], zoom_start=6)
for point in range(0, len(centers)):
    folium.Marker(centers[point]).add_to(map)
map


# In[ ]:


type(accident_data)


# In[ ]:


accident_data.head()


# In[ ]:


x = accident_data.sample(frac=.1)
# x = accident_data.head(1)
y = x[['Accident_Severity','Number_of_Vehicles','Number_of_Casualties','Weather_Conditions','Light_Conditions','Road_Surface_Conditions','Speed_limit']]
y['Weather_Conditions'] = y['Weather_Conditions'].astype('category')
y['Light_Conditions'] = y['Light_Conditions'].astype('category')
y['Road_Surface_Conditions'] = y['Road_Surface_Conditions'].astype('category')
y['Speed_limit'] = y['Speed_limit'].astype('category')

print(y.size)
y['Weather_Conditions_Cat'] = y['Weather_Conditions'].cat.codes
y['Light_Conditions_Cat'] = y['Light_Conditions'].cat.codes
y['Road_Surface_Conditions_Cat'] = y['Road_Surface_Conditions'].cat.codes
y['Speed_limit_Cat'] = y['Speed_limit'].cat.codes


y.head()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import seaborn as sb
sb.pairplot(y)


# In[ ]:


y.Weather_Conditions.value_counts()


# In[ ]:


# Excluding normal wether conditions, imply that biggest weather reason is Raining
pd.value_counts(accident_data['Weather_Conditions']).drop('Fine without high winds').plot.bar()


# In[ ]:


pd.value_counts(accident_data['Light_Conditions']).drop('Daylight: Street light present').plot.bar()
# pd.value_counts(accident_data['Light_Conditions']).plot.bar()


# In[ ]:


x.Speed_limit.value_counts()


# In[ ]:


x.shape


# In[ ]:




