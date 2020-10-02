#!/usr/bin/env python
# coding: utf-8

# # Not all Bars are Created Equally Loud#
# I was impressed with [Evgeniy Vasilev's Heatmap of pubs and bars](https://www.kaggle.com/somesnm/heatmap-of-pubs-and-bars-of-new-york-city) and wanted to do some work in that vein. I borrowed heavily from that and then continued on in another direction building a [choropleth](https://en.wikipedia.org/wiki/Choropleth_map) by number of complaints. I pulled a nyc neighboorhod geojson from [Pediacities](http://catalog.opendata.city/dataset/pediacities-nyc-neighborhoods) . I utilized GeoPandas to then combine neighborhood data with the noise complaint data using sjoin and dissolve with summation aggregation. Unfortunately rtree is not part of the environment so the code is commented out.

# In[17]:


from subprocess import check_output
print(check_output(["ls", "../input/nyc-neighborhoods"]).decode("utf8"))


# In[18]:


import numpy as np
import pandas as pd
import folium
from folium import features
from folium.plugins import HeatMap
from folium.plugins import MarkerCluster

import seaborn as sns
import time
import datetime
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[19]:


bars = pd.read_csv('../input/partynyc/bar_locations.csv')
bars.head()


# I used the median Lat and Long to center my maps.

# In[20]:


mlat = bars['Latitude'].median()
mlon = bars['Longitude'].median()
print(mlat, mlon)


# As I mentioned above the rtree library was not supported. In fact, it is a seperate install from geopandas that requires another C library to drive it. rtree though is what drives the dissolve function. The 

# In[21]:


'''
import geopandas as gpd
import json
import rtree
from shapely.geometry import Point

#Takes a geoJson and converts it into a GeoDataFrame
district_geo =r'hoods.json'
data2 = json.load(open(district_geo))
data3 = gpd.GeoDataFrame.from_features(data2['features'])
#Turns the bar data into a GeoDataFrame
bars['geometry']= [Point(xy) for xy in zip(bars.Longitude, bars.Latitude)]
crs = {'init': 'epsg:4326'}
gdf = gpd.GeoDataFrame(bars, crs=crs, geometry=bars['geometry'])
#Spatially joins the two GeoDataFrames determining if a bar is in a neighborhood
bars_with_hood = gpd.sjoin(gdf, data3, how="right", op='intersects')
#Sums each neighborhoods calls now that the bars have a neighborhood
calls_by_district = bars_with_hood.dissolve(by="neighborhood", aggfunc='sum')
calls_by_district.fillna(0, inplace=True)
#Creates a representative point in each neighborhood
calls_by_district['coords'] = calls_by_district['geometry'].apply(lambda x: x.representative_point().coords[:])
calls_by_district['coords'] = [coords[0] for coords in calls_by_district['coords']]
'''


# **The Choropleth**
# So here is what the aggregated calls by neighborhood looks like. In terms of policing or public transport staging, it becomes somewhat obvious where to put cars and officers. Chose to focus on complaints rather than sheer number of bars.

# In[22]:


#Neighborhood json
district_geo =r'../input/nyc-neighborhoods/hoods.json'

#pulls in the result from aggregation, indexes to neighborhood
calls_by_district = pd.read_csv('../input/nyc-neighborhoods/calls_by_district.csv') 
calls_by_district = calls_by_district.set_index('neighborhood')
calls =calls_by_district.iloc[:,5]

#build the map
map_test = folium.Map(location=[mlat, mlon], zoom_start=12)
map_test.choropleth(district_geo, data=calls, key_on='properties.neighborhood',
                    fill_color='YlGn',fill_opacity=0.7,line_opacity=0.2)


# In[23]:


#which neighborhood had the most calls?
trial = calls_by_district.nlargest(29, 'num_calls')
trial.head()


# In[24]:


map_test


# In[25]:


# X marks the spot.. well I gues O does here
for x in range (1,4):
    folium.CircleMarker([40.711662220917816, -73.9505630158169],
                        radius=x*10,
                        popup='Williamsburg',
                        color='Red',
                        fill_color='#3186cc',
                       ).add_to(map_test)
folium.Marker([40.711662220917816, -73.9505630158169],
                    popup='Williamsburg',
                   ).add_to(map_test)
for x in range (1,3):
    folium.CircleMarker([40.72629678410688, -73.98175826912397],
                        radius=x*10,
                        popup='East Village',
                        color='Red',
                        fill_color='#3186cc',
                       ).add_to(map_test)
folium.Marker([40.72629678410688, -73.98175826912397],
                    popup='East Village',
                   ).add_to(map_test)
map_test


# # Let's Zero In#
# What if we zoom into just Williamsburg and then try to isolate the best spot to stage transport and officers

# In[26]:


bars_with_hood = pd.read_csv('../input/nyc-neighborhoods/bars_with_hood.csv')
bars_with_hood.head()


# In[27]:


#Stats for the worst neighborhood
wb_bars=bars_with_hood.loc[bars_with_hood['neighborhood'] == 'Williamsburg']
wb_bars['num_calls'].describe()


# In[28]:


import re

# Let's Draw a another map and isolate the really loud bars
check = folium.Map(location=[40.711662220917816, -73.9505630158169], zoom_start=14)
#redrawing the neighborhood boundary
thing = calls_by_district.loc['Williamsburg']['geometry']
#Because this data was pulled in via CSV it lost the polygon datatype
#Used regex to pull the coordinates out of a string to make a new list
pat = re.compile(r'''(-*\d+\.\d+ -*\d+\.\d+);*''')
matches = pat.findall(thing)
if matches:
    lst = [tuple(map(float, m.split())) for m in matches]
thing= lst

#With new list in hand fed the coordinates to folium to draw the boundary
points = []
for x in range(0, len(thing)):
    points.append([thing[x][1],thing[x][0]])
folium.PolyLine(points, color="red", weight=2.5, opacity=1).add_to(check)
check


# ## One Standard Deviation of Noise##
# Now that the boundary was back in place, I wanted to isolate the really loud bars. Using the descriptive stats I put some logic into marker creation and cluster. If they were 1SD louder than the mean or greater they were not clustered and placed in red. If they were somewhere between the mean and 1SD they were placed in an orange icon and clustered if needed.

# In[30]:


mc = MarkerCluster()
for ind, row in wb_bars.iterrows():
    num = row['num_calls']
    if num > 71:
        num = str(num)
        folium.Marker(location=[row['Latitude'], row['Longitude']], popup=num,
                                   icon=folium.Icon(color='red',icon='info-sign')).add_to(check)
    elif num > 33 and num <=71:
        num = str(num)
        mc.add_child(folium.Marker(location=[row['Latitude'], row['Longitude']], popup=num,
                                   icon=folium.Icon(color='orange',icon='info-sign')))
check.add_child(mc)

folium.CircleMarker(location=[40.7178, -73.9577], line_color='Blue', fill_color='Blue', radius=20).add_to(check)
check


# Cool so now in theory, we should have the loudest bars in the dataset. What was interesting was that the northwest sector has only one major subway stop at Bedford Avenue, but a disproportionate number of loud bars. That makes me believe the premise that this really might all boil down to getting people in cabs or uber rides. Notice that there are no red bars (outlier level loud) near the station? There are a number of louder than normal bars in the area but none past 1SD. Of all of the red bars only one was on a major bus route. Several were somewhat close to subway stations but subways somewhat constrict travel options.

# In[ ]:




