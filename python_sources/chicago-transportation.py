#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# ### Note: this is very much a work-in-progress.
# 
# I am about to move to a different city in a few weeks. This is a very stressful time in terms of figuring out logistics of where to look for an apartment, how far it is going to be from a metro station and so on. I think I am not the only one for whom a lot of decisions depend on the availability of transportation. Chicago recently released data from ride-hailing apps such as Uber and Lyft. With the data for ride-hailing services, as well as taxi data, one can predict which neighborhoods are more likely to use such a service. This helps in two ways:
# 
# 1. For the government, this provides them with information on which neighborhoods may require improvements in terms of public transportation.
# 2. For people like me, this helps determine which regions of the city are more accessible with public transportations, or which regions have a lower cost associated with taxis or ride-hailing services.
# 
# All of the data is available at [Chicago Open Data](https://data.cityofchicago.org/)
# 
# I have used [this BigQuery dataset](https://www.kaggle.com/chicago/chicago-taxi-trips-bq) to create a working dataset for Chicago taxi data from 2017.
# 

# In[ ]:


import numpy as np 
import pandas as pd 
import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# # Rideshare Data
# 
# ## Importing and cleaning

# In[ ]:


get_ipython().run_cell_magic('time', '', "list_df = []\nchunksize = 10**6\n\nfor chunk in pd.read_csv('../input/chicago-rideshare/rideshare.csv',index_col=0,chunksize=chunksize):\n    df = chunk\n    list_df.append(df)")


# In[ ]:


rides = pd.concat(list_df)
rides.info()


# In[ ]:


rides.isnull().sum()


# We have a lot of data, so dropping the NaN should not be too bad.

# In[ ]:


rides.dropna(inplace=True)
rides.reset_index(drop=True,inplace=True)
rides.columns = rides.columns.str.replace(' ', '_').str.lower()


# In[ ]:


get_ipython().run_cell_magic('time', '', "rides['trip_start_timestamp'] =pd.to_datetime(rides['trip_start_timestamp'],format= '%m/%d/%Y %I:%M:%S %p')")


# Let's add the community names from the census data.

# In[ ]:


census = pd.read_csv('../input/chicago-census-socioecon-commarea-2012/Chicago_Census_SociaEcon_CommArea_2008_2012.csv')
census.columns = census.columns.str.replace(' ','_').str.lower()
census.columns


# In[ ]:


comm_dict = pd.Series(census.community_area_name,index=census.community_area_number).to_dict()
rides['pickup_community_area_name'] = rides['pickup_community_area'].map(lambda x: comm_dict[x])
rides['dropoff_community_area_name'] = rides['dropoff_community_area'].map(lambda x: comm_dict[x])


# Let's also add day of the week data and divide the day into different time chunks.

# In[ ]:


rides['weekday'] = rides['trip_start_timestamp'].map(lambda x: x.weekday())


# In[ ]:


rides['time'] = 0
rides.loc[rides['trip_start_timestamp'].dt.hour <= 4,'time'] = 5
rides.loc[(rides['trip_start_timestamp'].dt.hour > 4)&((rides['trip_start_timestamp'].dt.hour <= 8)),'time'] = 0
rides.loc[(rides['trip_start_timestamp'].dt.hour > 8)&((rides['trip_start_timestamp'].dt.hour <= 12)),'time'] = 1
rides.loc[(rides['trip_start_timestamp'].dt.hour > 12)&((rides['trip_start_timestamp'].dt.hour <= 16)),'time'] = 2
rides.loc[(rides['trip_start_timestamp'].dt.hour > 16)&((rides['trip_start_timestamp'].dt.hour <= 20)),'time'] = 3
rides.loc[(rides['trip_start_timestamp'].dt.hour > 20),'time'] = 4


# # Rides by time of day and day of the week

# In[ ]:


rides_by_time = rides.groupby('time')['trip_id'].count().reset_index(name = 'trips')
rides_by_day = rides.groupby('weekday')['trip_id'].count().reset_index(name = 'trips')


# In[ ]:


fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
rides_by_time.plot(kind='bar',y='trips',x='time',legend=None,figsize=(18,6),color=sns.color_palette('Blues_d'),ax=ax1)
ax1.set_xticklabels(labels= ["04:00-08:00","08:00-12:00", "12:00-16:00", "16:00-20:00", "20:00-00:00","00:00-04:00"],rotation=60)
ax1.set_xlabel('Time of Day')
ax1.set_ylabel('Number of Trips')
ax1.set_title('Rides by time of day')
rides_by_day.plot.bar(x='weekday',y='trips',legend=None,figsize=(18,6),ax=ax2,color=sns.color_palette('YlOrRd_d'))
ax2.set_xticklabels(labels= ["Monday", "Tuesday", "Wednesday","Thursday", "Friday","Saturday","Sunday"],rotation=60)
ax2.set_xlabel('Day of the Week')
ax2.set_ylabel('Number of Trips')
ax2.set_title('Rides by day of the week')
fig.subplots_adjust(wspace=0.3)
plt.savefig('rides_day_time.jpg')
fig.savefig('rides_day_time.pdf')


# 

# In[ ]:


sample = rides.sample(frac=0.1)
sample['hour'] = sample['trip_start_timestamp'].dt.hour
fig, ax= plt.subplots()
sns.boxplot(x='weekday',y='hour',data=sample)
ax.set_xticklabels(labels= ["Monday", "Tuesday", "Wednesday","Thursday", "Friday","Saturday","Sunday"],rotation=60)
ax.set_xlabel('Days')
ax.set_ylabel('Time of Day')
ax.set_title('Trip time boxplot by day')
fig.savefig('trip_time_boxplot.jpg')


# # Rides by community area:
# 
# Let's take a quick look the the map to see which community areas have a higher demand for ride-sharing servises.

# In[ ]:


import folium
from folium import IFrame, FeatureGroup, LayerControl, Map, Marker, plugins


# In[ ]:


dropoff_locations = rides.groupby('dropoff_community_area')[['dropoff_community_area_name','dropoff_centroid_latitude','dropoff_centroid_longitude']].first().reset_index()
pickup_locations = rides.groupby('pickup_community_area')[['pickup_community_area_name','pickup_centroid_latitude','pickup_centroid_longitude']].first().reset_index()


# In[ ]:


pickup_locations['trips'] = rides.groupby('pickup_community_area')['trip_id'].count().values
dropoff_locations['trips'] = rides.groupby('dropoff_community_area')['trip_id'].count().values


# In[ ]:


fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
pickup_locations.sort_values('trips',ascending=False).head(20).plot(kind='bar',y='trips',x='pickup_community_area_name',legend=None,figsize=(15,15),
                                                                    color=sns.cubehelix_palette(20,start=3,rot=-.25,reverse=True),ax=ax1)
#ax1.set_xticklabels(labels= ["04:00-08:00","08:00-12:00", "12:00-16:00", "16:00-20:00", "20:00-00:00","00:00-04:00"],rotation=60)
ax1.set_xlabel('Community Area')
ax1.set_ylabel('Number of Pickups')
ax1.set_title('Most Rides by Community Area (Pickups)')
dropoff_locations.sort_values('trips',ascending=False).head(20).plot(kind='bar',x='dropoff_community_area_name',y='trips',legend=None,figsize=(15,15),ax=ax2,
                                                                     color=sns.cubehelix_palette(20,reverse=True))
#ax2.set_xticklabels(labels= ["Monday", "Tuesday", "Wednesday","Thursday", "Friday","Saturday","Sunday"],rotation=60)
ax2.set_xlabel('Community Area')
ax2.set_ylabel('Number of Trips')
ax2.set_title('Most Rides by Community Area (Dropoffs)')
fig.subplots_adjust(hspace=0.5)
plt.savefig('rides_by_area.jpg')
fig.savefig('rides_by_area.pdf')


# In[ ]:


pickup_locations.head()


# # Interactive Map of Trips by Community Area

# In[ ]:


Chicago_COORDINATES = (41.895140898, -87.624255632)
chicago_pickup_map = folium.Map(location=Chicago_COORDINATES,
                        zoom_start=11,tiles='CartoDB positron')


for i in range(len(pickup_locations)):
    lat = pickup_locations.iloc[i][2]
    long = pickup_locations.iloc[i][3]
    radius = 1.5*np.log(pickup_locations['trips'].iloc[i])
    if pickup_locations['trips'].iloc[i] >= 10**6:
        color = 'navy'
    elif pickup_locations['trips'].iloc[i] >=10**5:
        color = 'royalblue'
    else:
        color = 'lightseagreen'
    popup_text = """Community Area : {}<br>
                Pickups : {}<br>"""
    popup_text = popup_text.format(pickup_locations.iloc[i][1],
                              pickup_locations['trips'].iloc[i])
    folium.CircleMarker(location = [lat, long], radius=radius,popup= popup_text,color=color, fill = True).add_to(chicago_pickup_map)


# In[ ]:


chicago_pickup_map.save('chicago_pickups_html')


# In[ ]:


chicago_dropoff_map = folium.Map(location=Chicago_COORDINATES,
                        zoom_start=11,tiles='CartoDB positron')


for i in range(len(dropoff_locations)):
    lat = dropoff_locations.iloc[i][2]
    long = dropoff_locations.iloc[i][3]
    radius = 1.5*np.log(dropoff_locations['trips'].iloc[i])
    if dropoff_locations['trips'].iloc[i] >= 10**6:
        color = 'darkred'
    elif dropoff_locations['trips'].iloc[i] >= 10**5:
        color='red'
    else:
        color = 'salmon'
    popup_text = """Community Area : {}<br>
                Dropoffs : {}<br>"""
    popup_text = popup_text.format(dropoff_locations.iloc[i][1],
                              dropoff_locations['trips'].iloc[i])
    folium.CircleMarker(location = [lat, long], radius=radius,popup= popup_text,color=color, fill = True).add_to(chicago_dropoff_map)


# In[ ]:


chicago_dropoff_map.save('chicago_dropoff.html')


# In[ ]:


chmap = folium.Map(location=Chicago_COORDINATES, zoom_start=11,tiles='CartoDB positron')
for i in range(len(pickup_locations)):
    lat = pickup_locations.iloc[i][2]
    long = pickup_locations.iloc[i][3]
    radius = 1.5*np.log(pickup_locations['trips'].iloc[i])
    if pickup_locations['trips'].iloc[i] >= 10**6:
        color = 'navy'
    elif pickup_locations['trips'].iloc[i] >=10**5:
        color = 'royalblue'
    else:
        color = 'lightseagreen'
    popup_text = """Community Area : {}<br>
                Pickups : {}<br>"""
    popup_text = popup_text.format(pickup_locations.iloc[i][1],
                              pickup_locations['trips'].iloc[i])
    folium.CircleMarker(location = [lat, long], radius=radius,popup= popup_text,color=color, fill = True).add_to(chmap)

for i in range(len(dropoff_locations)):
    lat = dropoff_locations.iloc[i][2]
    long = dropoff_locations.iloc[i][3]
    radius = 1.5*np.log(dropoff_locations['trips'].iloc[i])
    if dropoff_locations['trips'].iloc[i] >= 10**6:
        color = 'darkred'
    elif dropoff_locations['trips'].iloc[i] >= 10**5:
        color='red'
    else:
        color = 'salmon'
    popup_text = """Community Area : {}<br>
                Dropoffs : {}<br>"""
    popup_text = popup_text.format(dropoff_locations.iloc[i][1],
                              dropoff_locations['trips'].iloc[i])
    folium.CircleMarker(location = [lat, long], radius=radius,popup= popup_text,color=color, fill = True).add_to(chmap)


# In[ ]:


chmap


# In[ ]:


chmap.save('chicago_rides.html')


# Of course, this is only preliminary analysis. There is much more to be done.
# 1. One would get a clearer picture after considering some other factors such as population in each area and whether or not the areas have tourist attractions and workplaces.
# 
# 2. The census data serves the purpose of taking into consideration that some citizens are comparatively well-off financially and are therefore more likely to avoid public transportation and choose the comfort of a ride-hailing service. 
# 
# 3. Another interesting thing to note would be the length and fare of the rides and frequencies of those cross referenced against area, time of day and day of week. For example, rides that take about twenty minutes or thirty minutes or so are more likely to be avoided if there is reliable public transportation.
# 
# 4. One could look at variation in fares for rides of similar length based on pickup and dropoff locations.

#  # Taxi data
# 
# This is a chunk of the taxi data from 2017. One could do a similar analysis with taxi data and combine both two draw conclusions.

# In[ ]:


taxi = pd.read_csv('../input/chicago-taxi-2017/taxi2017.csv',index_col=0)
taxi.head()


# In[ ]:




