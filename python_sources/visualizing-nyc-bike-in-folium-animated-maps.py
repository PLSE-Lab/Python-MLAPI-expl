#!/usr/bin/env python
# coding: utf-8

# **Heads up:** This notebook has the code of the article published in LinkedIn: 
# If you want to get a deeper explanation of each cell, I recommend you to read the article [here](https://medium.com/p/visualizing-nyc-bike-data-on-interactive-and-animated-maps-with-folium-plugins-c2d7645cd19b?source=email-9cb2a72cad71--writer.postDistributed&sk=d2237ce96bd8a490ac4e3dc7029fd8bb). 
# 
# The code is also available for download on GitHub [here](https://github.com/Bondify/Python/blob/master/Visualizing%20NYC%20Bike%20Data%20on%20interactive%20and%20animated%20maps%20with%20Folium%20plugins.ipynb).

# In[ ]:


# Import the libraries
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
import datetime


# In[ ]:


# Create a folium map as an example
folium.Map(location = [40.71958611647166, -74.0431174635887],
           tiles='cartodbpositron',
           zoom_start=12)


# In[ ]:


# Importing data
nyc = pd.read_csv('../input/nyc_data.csv')
nyc.head(2)


# In[ ]:


# Check the type of the 'starttime variable'
type(nyc.loc[0,'starttime'])


# In[ ]:


# Setting the right format for starttime and stoptime
nyc['starttime'] = nyc['starttime'].str[:-5]
nyc['stoptime'] = nyc['stoptime'].str[:-5]
nyc['starttime'] = pd.to_datetime(nyc['starttime'])
nyc['stoptime'] = pd.to_datetime(nyc['stoptime'])

# Define the startime as index
nyc = nyc.set_index('starttime')
nyc['type'] = 'station'
nyc.head(1)


# # TimeStampedGeoJson

# In[ ]:


# Aggregate number of trips for each start station by hour of the day
start = nyc.pivot_table('tripduration', 
                     index = ['start station id', 
                              'start station latitude', 
                              'start station longitude', 
                              nyc.index.hour],
                     columns = 'type',
                     aggfunc='count').reset_index()

start.head()


# In[ ]:


# Get the daily average
days = nyc.index.day.max()
start['station'] = start['station']/days


# In[ ]:


# Change the name of the columns
start.columns = ['station_id', 'lat', 'lon', 'hour', 'count']
# Create a color variable for mapping
start['fillColor'] = '#53c688'
# Change the color for those stations with less than one daily trip
start.loc[start['count']<1, 'fillColor'] = '#586065'
start.head(1)


# In[ ]:


# Create the geojson with the right format to use in the folium plugin
import datetime
def create_geojson_features(df):
    features = []
    
    for _, row in df.iterrows():
        feature = {
            'type': 'Feature',
            'geometry': {
                'type':'Point', 
                'coordinates':[row['lon'],row['lat']]
            },
            'properties': {
                'time': pd.to_datetime(row['hour'], unit='h').__str__(),
                'style': {'color' : ''},
                'icon': 'circle',
                'iconstyle':{
                    'fillColor': row['fillColor'],
                    'fillOpacity': 0.8,
                    'stroke': 'true',
                    'radius': row['count'] + 5
                }
            }
        }
        features.append(feature)
    return features

start_geojson = create_geojson_features(start)
start_geojson[0]


# In[ ]:


# Create the map and animate the data
from folium.plugins import TimestampedGeoJson

nyc_map = folium.Map(location = [40.71958611647166, -74.0431174635887],
                    tiles = "CartoDB Positron",
                    zoom_start = 14)

TimestampedGeoJson(start_geojson,
                  period = 'PT1H',
                  duration = 'PT1M',
                  transition_time = 1000,
                  auto_play = True).add_to(nyc_map)
nyc_map


# # DualMap

# In[ ]:


# Do the same as before in the opposite direction to have both in a dual map

nyc1= nyc.reset_index().set_index('stoptime')
end = nyc1.pivot_table('tripduration', 
                     index = ['end station id', 
                              'end station latitude', 
                              'end station longitude', 
                              nyc1.index.hour],
                     columns = 'type',
                     aggfunc='count').reset_index()

end['station'] = end['station']/days

end.columns = ['station_id', 'lat', 'lon', 'hour', 'count']
end['fillColor'] = '#e64c4e'
end.loc[end['count']<1, 'fillColor'] = '#586065'


# In[ ]:


# Create the geojson with the right format to animate 
end_geojson = create_geojson_features(end)


# In[ ]:


# Create the dual map with both directions animated
from folium.plugins import DualMap

dualmap = DualMap(location = [40.71958611647166, -74.0431174635887],
                 tiles = 'cartodbpositron',
                 zoom_start = 14)

TimestampedGeoJson(start_geojson,
                  period = 'PT1H',
                  duration = 'PT1M',
                  transition_time = 250,
                  auto_play = True).add_to(dualmap.m1)

TimestampedGeoJson(end_geojson,
                  period = 'PT1H',
                  duration = 'PT1M',
                  transition_time = 250,
                  auto_play = True).add_to(dualmap.m2)

dualmap


# # AntPath

# In[ ]:


# Get the stations with more interactions
df = nyc.pivot_table('tripduration', 
                     index = ['start station name', 
                              'end station name',],
                     columns = ['type'],
                     aggfunc='count').reset_index().sort_values(by='station', ascending=False)
df.head()


# In[ ]:


# Take the top stations as a mask to filter data afterwards
mask1 = df["start station name"].head(10)
mask2 = df["end station name"].head(10)
mask = mask1.append(mask2)
mask = mask.unique()
mask


# In[ ]:


# Get the peak hours
nyc['time_of_day'] = pd.cut(nyc.index.hour,
                            [0,6,10,16,20,24],
                            labels = ['am_valley', 'am_peak', 'mid_valley', 'pm_peak', 'pm_valley'], 
                            right=False)

am = nyc.loc[nyc['time_of_day'] == 'am_peak', :]
pm = nyc.loc[nyc['time_of_day'] == 'pm_peak', :]


# In[ ]:


# We'll take the station 'Grove St PATH and analyze it's commutes patterns 
# in the morning and the afternoon

to_st_path = am.loc[(am['end station name'] == 'Grove St PATH') & (am['start station name'].isin(mask)), :]
from_st_path = pm.loc[(pm['start station name'] == 'Grove St PATH') & (pm['end station name'].isin(mask)), :]

to_st_path.head(1)


# In[ ]:


to_st_path = to_st_path.pivot_table('tripduration',
                                   index = ['start station name', 
                                            'start station latitude',
                                            'start station longitude',
                                            'end station name',
                                            'end station latitude',
                                            'end station longitude'],
                                   columns = ['type'],
                                   aggfunc='count').reset_index().sort_values(by='station', ascending=False)

to_st_path


# In[ ]:


from_st_path = from_st_path.pivot_table('tripduration',
                                   index = ['start station name', 
                                            'start station latitude',
                                            'start station longitude',
                                            'end station name',
                                            'end station latitude',
                                            'end station longitude'],
                                   columns = ['type'],
                                   aggfunc='count').reset_index().sort_values(by='station', ascending=False)

from_st_path


# In[ ]:


# The weight scale will be from 1-10 
# 365 --> 10
# 124 --> 1

# 365station/10weight=124station/1weight --> 1weight=124station*10weight/365station
# 265*10/365= 7.26

# The delay has to be higher for lower numbers
# To
# y=a*x + b --> x=628 --> y=100 --> 100=628*a + b --> b = 100 - 628*a --> b = 
# y=a*x + b --> 800=188*a + 100 - 628*a --> 700 = (188 - 628)*a = -440*a --> a=-700/440 --> a = -2.9

# From
# y=a*x + b --> x=365 --> y=100 --> 100=365*a + b --> b = 100 - 365*a --> b = 1160
# y=a*x + b --> 800=124*a + 100 - 365*a --> 700 = (124 - 365)*a = -241*a --> a=-700/241 --> a = -2.9

to_st_path['weight'] = to_st_path['station']*10/(to_st_path['station'].max())
from_st_path['weight'] = from_st_path['station']*10/(from_st_path['station'].max())

a_to_st_path = (800-100)/(to_st_path['station'].min() - to_st_path['station'].max())
b_to_st_path = 100 - to_st_path['station'].max()*a_to_st_path

a_from_st_path = (800-100)/(from_st_path['station'].min() - from_st_path['station'].max())
b_from_st_path = 100 - from_st_path['station'].max()*a_from_st_path

to_st_path['delay'] = a_to_st_path*to_st_path['station'] + b_to_st_path
from_st_path['delay'] = a_from_st_path*from_st_path['station'] + b_from_st_path


# In[ ]:


from folium.plugins import DualMap

dualmap = DualMap(location = [to_st_path.loc[0, 'end station latitude'], to_st_path.loc[0, 'end station longitude']],
                         tiles='cartodbpositron',
                         zoom_start=15)

to_st_path.apply(lambda row: folium.plugins.AntPath([(row['start station latitude'],
                                                      row['start station longitude']),
                                                     (row['end station latitude'],
                                                      row['end station longitude'])],
                                                    color='blue',
                                                    weight = row['weight'],
                                                    delay = row['delay']).add_to(dualmap.m1),
                   axis=1)

from_st_path.apply(lambda row: folium.plugins.AntPath([(row['start station latitude'],
                                                        row['start station longitude']),
                                                       (row['end station latitude'],
                                                        row['end station longitude'])],
                                                      color='blue',
                                                      weight = row['weight'],
                                                      delay = row['delay']).add_to(dualmap.m2),
                   axis=1)

dualmap


# # Heatmap

# In[ ]:


# Create the list of lists needed to use the plugin
# Check the documentation for more details
df_hour_list = []
hours = pd.Series(nyc.index.hour.unique().sort_values())

def create_list(hour):
    df_hour_list.append(nyc.loc[nyc.index.hour == hour,
                                ['start station latitude',
                                 'start station longitude']].
                        groupby(['start station latitude', 
                                 'start station longitude']).sum().reset_index().values.tolist())
    
hours.apply(create_list);


# In[ ]:


# This map shows an animated view of the stops with their radius depending on the total amount of trips
# and a dynamic view of the events that take place as a heatmap

from folium.plugins import HeatMapWithTime

# Add trip events to the map
map_time = folium.Map(location=[40.71958611647166, -74.0431174635887], tiles="CartoDB Positron", zoom_start=14)

HeatMapWithTime(df_hour_list, 
                auto_play=True, 
                max_opacity=0.5, 
                gradient = {0.2: '#FBD973', 0.4: '#fa782f', 0.75: '#F16578', 1: '#782890'}).add_to(map_time)

map_time


# In[ ]:




