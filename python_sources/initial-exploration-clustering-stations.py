#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# #  Initial Exploration
# 
# * Read Dataset
# * Visualization
# * Stations Similarity - Clustering

# In[2]:


import pandas as pd
import os
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import datetime as dt

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('white')


# ### Read Dataset

# In[3]:


df = pd.read_csv('../input/sp_bike.csv')
df.head()


# In[4]:


#  Transform 
df['created_at'] = pd.to_datetime(df.created_at)

# WeekDay
df['weekday'] = df['created_at'].dt.weekday

# Time
df['hour']     = df['created_at'].apply(lambda x: dt.datetime.strftime(x, '%H'))

df['hour_min'] = df['created_at'].apply(lambda x: dt.datetime.strftime(x, '%H:%M'))

# Create field #completude. 
#   Each station can contain different sizes, more or less bicycles. The #completute is the value normalized about the total bicycles off
#   #completude is in 0..100%
df['completude'] = df.available/(df.available + df.free)


# In[5]:


# Filter Inactive Statins
df = df[df.status == 'A']


# In[6]:


df.head()


# In[7]:


# Filter only unique stations in dadaset
df_stations  = df[['lat', 'lng', 'name', 'address']].drop_duplicates()
df_stations.head()


# In[8]:


print("Total Stations: ", len(df_stations))


# ### Visualization

# In[9]:


import folium


# In[10]:


# Plot Stations in Map 
#

# Center Cordinate
cord_center = [df_stations.lat.mean(), df_stations.lng.mean()]

# Map
mp = folium.Map(location=cord_center, zoom_start=14, tiles='cartodbpositron')

# Plot
for i, location in df_stations.iterrows():
    folium.CircleMarker(
        location=[location['lat'],location['lng']],
        radius=4,
        popup=location['name']
    ).add_to(mp)
mp


# In[11]:


# Stations
df_stations.head(5)


# In[12]:


# Filter dataset only one stations
df_temp = df[df.name == df_stations.iloc[0]['name']]


# Plot the evolution of the completeness of this stations at the time of day.  Separete on the day of the week,  because I hope the weekend is different pattern or another day.

# In[13]:


def plt_completude_in_time(datasets, label = []):
    '''
    Plot TimeSerie of Completude.
    '''
    fig, ax = plt.subplots(figsize=(15,6))
    plt.title("Evolution of completeness", loc='left')
    plt.xlabel('Time')
    plt.ylabel("% full")    
    
    for i in range(len(datasets)):
        df = datasets[i]
        time_avg = df.groupby('hour', as_index=False).agg('mean')
        plt.plot(time_avg['hour'], time_avg['completude'], label=label[i])
        plt.legend()
        
    sns.despine()

days = {0:'Mon',1:'Tues',2:'Weds',3:'Thurs',4:'Fri',5:'Sat', 6:'Sun'}

plt_completude_in_time(
    [df_temp[df_temp.weekday == w] for w in range(7)],
    [days[w] for w in range(7)])    


# In this station, Tues at Fri the completude is the same pattern, full between 00 at 07h and decays until approximately 18h. The Monday is diferrent, i beleave that nobody wants use bicicle to go work.  The Sunday, the completude is less 50% everytime, i belieave that everyone wants use the bicicle.
# 
# 

# #### Use bicycle by time

# In[17]:


df_week     = df[df.weekday < 5]
df_weekend  = df[df.weekday >= 5]

df_week    = df_week.groupby('hour', as_index=False).agg('mean')
df_weekend = df_weekend.groupby('hour', as_index=False).agg('mean')

#Plot cluster centers
'''
Plot TimeSerie of Use of bicycles Available per Hour.
'''
fig, ax = plt.subplots(figsize=(15,6))
plt.title("Use of bicycles by time", loc='left')
plt.xlabel('Time')
plt.ylabel("Use of bicycles")    

# PLot Week
plt.plot(df_week.hour, 1-df_week.completude, color='red', label='Week')
plt.legend()

#Plot Weekend
plt.plot(df_weekend.hour, 1-df_weekend.completude, color='blue', label='Weekend')
plt.legend()

sns.despine()


# Makes sense. In the week because of the workload, between 8am to 18h the used bicycles are few, having peaks in the hours of going to work (08h) and back (18h). Weekend, the use is lasting all day. 

# ### Stations Similarity - Clustering
# 
# I am grouping the stations with the seventy idea:
# 
# * Full stations during business hours
# * Empy stations during business hours
# * Others
# 
# This clustering can give an idea of the flow of use of the stations to go to work.

# In[18]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


# In[19]:


df_temp  = df[df.weekday < 5]
df_temp.groupby('hour_min', as_index=False).agg('mean').head()


# In[31]:


# Filter only week interactions
df_temp  = df[df.weekday.astype(int) < 5]

# Filter features by station
data = []
for i, location in df_stations.iterrows():
    df_temp_s  = df_temp[df_temp['name'] == location['name']].groupby(['hour']).agg('mean')
    features   = df_temp_s.completude.values.T
    #if len(features) == 154:
    data.append(features)
    #print(len(features)) 1-df_week.completude


# In[32]:


print("Station '", df_stations.iloc[0]['name'], "' features: \n", data[0], "\n", len(data[0]))


# In[ ]:





# Features is the mean of completude in hout of day.

# In[33]:


# Normalize Features
scaler = MinMaxScaler()
n_data = scaler.fit_transform(data)
n_data = data#/data.max()


# In[34]:


# Clustering with Kmeans
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters).fit(n_data)


# In[40]:


#Plot cluster centers
colours = ['red','blue','green']

'''
Plot TimeSerie of Completude.
'''
fig, ax = plt.subplots(figsize=(15,6))
plt.title("Evolution of completeness", loc='left')
plt.xlabel('Time')
plt.ylabel("% full")    
#ax.set_ylim([0,100])

for k, colour in zip(kmeans.cluster_centers_, colours):
    plt.plot(k, color=colour,label=colour)
sns.despine()


# In[ ]:





# In[ ]:




