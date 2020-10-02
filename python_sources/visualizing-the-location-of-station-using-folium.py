#!/usr/bin/env python
# coding: utf-8

# # Purpose
# 
# This dataset measures the amount of air pollutants(SO2, NO2, CO, O3, PM10 and PM2.5) every hour through air pollution equipment in Seoul, Korea.  
# In this kernel, I simply use the Folium to see where the air pollution equipment is located in Seoul. Also, I will try to visualize where there are many and few air pollutants at a certain time.
# 
# # What is Folium
# 
# See [this kernel](1) for a description of Folium.
# 
# Folium is a powerful Python library that helps you create several types of Leaflet maps. The fact that the Folium results are interactive makes this library very useful for dashboard building. 
#   
# Folium was developed for the sole purpose of visualizing geospatial data.
# 
# [1]: https://www.kaggle.com/mrushan3/data-visualization-with-folium-library/comments#722403

# # import library

# In[ ]:


import pandas as pd 
import random

# for visualization
import folium
import json


# # Load data
# As you can see from the description of the dataset, it provides four data files.
#     
# This kernel will use the following data:
# - `Measure_summary.csv` with a summary of information
# - `Measurement_item_info.csv` with information on pollutants
# - `Measurement_station_info.csv` with information about measuring station 
# 

# In[ ]:


# load data
df = pd.read_csv('../input/air-pollution-in-seoul/AirPollutionSeoul/Measurement_summary.csv')
df_item = pd.read_csv('../input/air-pollution-in-seoul/AirPollutionSeoul/Original Data/Measurement_item_info.csv')
df_station = pd.read_csv('../input/air-pollution-in-seoul/AirPollutionSeoul/Original Data/Measurement_station_info.csv')


# # Visualization
# 
# ## Check the location of the measuring station
# Seoul is divided into 25 gu(district). To visualize Seoul by dividing it into different districts, we need [a json file](1) containing information on districts.  
#   
# 
# Using `Latitude` and `Longitude` stored in `Measurement_station_info.csv`, the location of the monitoring station is marked with a marker.
# 
# 
# [1]: https://raw.githubusercontent.com/southkorea/seoul-maps/master/kostat/2013/json/seoul_municipalities_geo_simple.json
# 

# In[ ]:



center = [37.541, 126.986] # center of Seoul
m = folium.Map(location=center, zoom_start=11) # set map

# load district information
with open('../input/seoul-geo-json/seoul_municipalities_geo.json',mode='rt',encoding='utf-8') as f:
    geo = json.loads(f.read())
    f.close()

# Add geojson to folium
folium.GeoJson(
    geo,
    name='seoul_municipalities'
).add_to(m)

# Add marker
for i in df_station.index[:25]: 
    popup_str = 'Station ' + str(df_station.loc[i, 'Station code'])
    folium.Marker(df_station.loc[i, ['Latitude', 'Longitude']],
                  popup=popup_str,
                  icon=folium.Icon(color='black')).add_to(m)

m # print


# As you can see on the map above, Seoul is divided into 25 districts.  
# The location of the measuring stations also shows that one measuring station is located in each districts.

# ## Markers according to status
# 
# `Measurement_item_info.csv` tells us what the current status is based on the value of the measured item.
# - Good (Blue)
# - Normal (Green)
# - Bad (Yellow)
# - Very bad (Red)  
# 
# Use these values to visualize the status of different regions by different colors of markers.  
# 
# (This criteria value may vary from country to country because it is the criteria for air pollutants in Seoul)
#  

# In[ ]:


def get_criteria(df_item, item):
    criteria = df_item[df_item['Item name'] == item].iloc[0, 3:]
    return criteria

def seoulmap(df_day, df_item, item):
    criteria = get_criteria(df_item, item)
    
    dfm = df_day.copy()
    
    # set color of marker
    dfm['color'] = ''
    dfm.loc[dfm[item] <= criteria[3], 'color'] = 'red'
    dfm.loc[dfm[item] <= criteria[2], 'color'] = 'orange' # yellow
    dfm.loc[dfm[item] <= criteria[1], 'color'] = 'green'
    dfm.loc[dfm[item] <= criteria[0], 'color'] = 'blue'
    
    center = [37.541, 126.986] # center of Seoul
    m = folium.Map(location=center, zoom_start=11) # set map

    with open('../input/seoul-geo-json/seoul_municipalities_geo.json',mode='rt',encoding='utf-8') as f:
        geo = json.loads(f.read())
        f.close()

    folium.GeoJson(
        geo,
        name='seoul_municipalities'
    ).add_to(m)

    for i in dfm.index: 
        popup_str = 'Station ' + str(dfm.loc[i, 'Station code']) + ': ' + str(dfm.loc[i, item])
        folium.Marker(dfm.loc[i, ['Latitude', 'Longitude']],
                      popup=popup_str,
                      icon=folium.Icon(color=dfm.loc[i, 'color'])).add_to(m)
    
    return m


# In[ ]:


random.seed(0)
ind = random.randint(1, len(df))

day = df.loc[ind, 'Measurement date']
print(day)
df_day = df[df['Measurement date'] == day]

seoulmap(df_day, df_item, 'PM10')


# In[ ]:


seoulmap(df_day, df_item, 'PM2.5')


# In[ ]:


random.seed(1)
ind = random.randint(1, len(df))

day = df.loc[ind, 'Measurement date']
print(day)
df_day = df[df['Measurement date'] == day]

seoulmap(df_day, df_item, 'PM10')


# In[ ]:


seoulmap(df_day, df_item, 'PM2.5')

