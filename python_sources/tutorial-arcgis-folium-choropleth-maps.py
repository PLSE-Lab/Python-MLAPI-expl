#!/usr/bin/env python
# coding: utf-8

# #  Tutorial:  ArcGIS, Folium and Choropleth Maps<a d="top"></a>
# ---

# This is a short tutorial/quick start guide on how to obtain location data using **GeoCoder-ArcGIS**, create interactive   and Choropleth maps using **Folium**.<br>
# 
# <div class="alert alert-block alert-warning">
# <b>NOTE:</b> GeoCoder is not accessable from Kaggle.  All code in ArcGIS section has been commented out.  I'm leaving the code in there so you may use it in another IDE.  I have provided the output screenshots from my IDE.
# </div><br>
# 
# 
#   -  **GeoCode-ArcGIS** is a great utility in geocoder for obtaining the latitude and longitude  of a location.   
# website:  https://www.esri.com/en-us/arcgis/about-arcgis/overview
# <br><br>
#   -  **Folium** is a Python library for creating interactive maps.   
# website:  https://python-visualization.github.io/folium/
# 
# 
# ## Table Of Content
# 1. [ArcGIS](#arc) - *not available on KAGGLE*<br>
# 1.1  [Lookup Latitude and Longitude with ArcGIS](#arc_search)<br>
# 1.2  [Define ArcGIS Function](#arc_func)<br>
# 1.3  [Execute ArcGIS Function](#arc_exec)<br>
# 1.4  [Invalid Location and ArcGIS](#arc_invalid)<br>
# 2. [Folium Maps](#fol)<br>
#       2.1 [World Map](#fol_world)<br>
#       2.2 [Markers](#fol_marker)<br>
# 3. [Choropleth Maps](#choro)<br>
# 3.1  [Create Dataframe](#choro_df)<br>
# 3.2  [Read geoJSON File](#choro_json)<br>
# 3.3  [Create Choropleth Map](#choro_plot)<br>
# 
# 
# -  - -
#  * **Lat/Lng Conversion**:  Folium works with **decimal** values for Latitude and Longitude.  Use the formula below if Latitude and Longitude are only available in degree/minutes/seconds:
#     
#     `Latitude  (dec) = degrees + (minutes/60) + (seconds/3600)`<br>
#     `Longitude (dec) = degrees + (minutes/60) + (seconds/3600)`
# <br>
# 
#  * **Install Folium**:  Jupyter supports folium.  For other applications/IDE (e.g. PyCharm, Spyder, Sublime, etc.), use "pip install folium --upgrade" or respective "install" variations.
# 
# <br>
# I will keep adding more info/code as I go along.  Upvote if you found this useful :-)

# ###   Import Libraries

# In[ ]:


import pandas as pd
import warnings
warnings.filterwarnings('ignore')

#  Kaggle directories
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#import geocoder   #  geocoder with ArcGIS  <- NOT AVAILABLE ON KAGGLE
import folium      #  folium libraries
from   folium.plugins import MarkerCluster


# [top](#top)
# 
# ---
# #  1.  GeoCoder - ArcGIS<a id="arc"></a>
# 
# <div class="alert alert-block alert-warning">
# <b>NOTE:</b> GeoCoder is not accessable from Kaggle.  All code in ArcGIS section has been commented out.  I'm leaving the code in there so you may use it in another IDE.
# </div><br>

# ###  1.1  Lookup Latitude and Longitude with ArcGIS<a id="arc_search"></a>
# Use **".latlng"** extension with **geocoder.arcgis** to get the latitude and longitude.

# In[ ]:


#  geocoder.arcgis('dallas, texas').latlng    # get lat & lng coord


# ###  1.2  Define ArcGIS Function<a id="arc_func"></a>
# Define function to get latitude and longitude.

# In[ ]:


'''
def arc_latlng(location):
    g = geocoder.arcgis('{}'.format(location))
    lat_lng_coords = g.latlng
    print(location,lat_lng_coords)
    return lat_lng_coords

arc_latlng('dallas, texas')     #  test arc_latlng
'''


# **OUTPUT**
# ![arcGis1.PNG](attachment:arcGis1.PNG)

# ###  1.3  Execute ArcGIS Function<a id="arc_exec"></a>
# Use the `get_latlng` function to get the latitude and longitude of several locations and save the information in a dataframe.  ARCGIS can take location names, zip/postal codes, landmark, etc.   
# <br>
# Sometimes, the dataframe columns do not maintain their order.  Verify that the column order is correct after dataframe is created.

# In[ ]:


'''
#  location list
#  10001 is the zip code of Manhattan, New York, US
#  M9B   is a postal code in Toronto, Canada
#  Everest is Mt. Everest in Nepal
location = ['10001','Tokyo','Sydney','Beijing','Karachi','Dehli', 'Everest','M9B','Eiffel Tower','Sao Paulo','Moscow']


#  call get_latlng function
loc_latlng = [arc_latlng(location) for location in location]


#  create dataframe for the results
df = pd.DataFrame(data = loc_latlng, columns = {'Latitude','Longitude'})
df.columns = ['Latitude','Longitude']  #  correct column order
df['Location'] = location              #  add location names
'''


# **OUTPUT**
# ![arcGis2.PNG](attachment:arcGis2.PNG)

# ###  1.4  Invalid Location and ArcGIS<a id="arc_invalid"></a>
# Invalid location will return a ***"None"*** value and will throw an error when creating the dataframe.

# In[ ]:


'''
invalid_loc = ['london','berlin','0902iuey7','999paris']  # 3 & 4 are invalid
invalid_latlng = [arc_latlng(invalid_loc) for invalid_loc in invalid_loc]
'''


# **OUTPUT**
# ![arcGis3.PNG](attachment:arcGis3.PNG)

# *Again, dataframe creation will fail because of the* "None".

# [top](#top)
# 
# ---
# #  2.  Folium Maps <a id="fol"></a>

# ###  2.1  World Map<a id="fol_world"></a>
# Create dataframe with location, latitude and longitude.
# 
# NOTE:  **geocoder** is not available on Kaggle.  Dataframe will be created manually.

# In[ ]:


Location  = ['10001', 'Tokyo', 'Sydney', 'Beijing', 'Karachi', 'Dehli', 'Everest', 'M9B', 'Eiffel Tower', 'Sao Paulo', 'Moscow']
Latitude  = [40.74876000000006, 35.68945633200008, -33.869599999999934, 39.90750000000003, 24.90560000000005, 28.653810000000078, 27.987910000000056, 43.64969222700006, 48.85859991892235, -23.562869999999975, 55.75696000000005]
Longitude = [-73.99331999999998, 139.69171608500005, 151.2069100000001, 116.39723000000004, 67.08220000000006, 77.22897000000006, 86.92529000000007, -79.55394499999994, 2.293980070546176, -46.654679999999985, 37.61502000000007]

df = pd.DataFrame(columns = {'Location','Latitude','Longitude'})
df.columns = ['Location','Latitude','Longitude']
df['Location']  = Location
df['Latitude']  = Latitude
df['Longitude'] = Longitude

df


# Note the following components in the Folium map:
# 
#   *  **location** - lat/lng can be any valid value
#   *  **tiles** - style of map
#   *  **zoom_start** - higher the number, closer the zoom
#   *  **popup** - text when a marker is clicked

# In[ ]:


#  center map on mean of Latitude/Longitude
map_world = folium.Map(location=[df.Latitude.mean(), df.Longitude.mean()], tiles = 'stamenterrain', zoom_start = 2)

#  add Locations to map
for lat, lng, label in zip(df.Latitude, df.Longitude, df.Location):
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        fill=True,
        color='Blue',
        fill_color='Yellow',
        fill_opacity=0.6
        ).add_to(map_world)

#  display interactive map
map_world

#  save map to local machine, open in any browser
#  map_world.save("C:\\ ... <path> ... \map_world_NYC.html")


# ###  2.1  Markers<a id="fol_mark"></a>
# **CircleMarker** and **MarkerCluster** are two of the most common markers in Folium.  CircleMarker was used in the previous example.  Let's create a dataframe to use with MarkerCluster with locations around New York City.

# In[ ]:


LocationNY  = ['Empire State Building', 'Central Park', 'Wall Street', 'Brooklyn Bridge', 'Statue of Liberty', 'Rockefeller Center', 'Guggenheim Museum', 'Metlife Building', 'Times Square', 'United Nations Headquarters', 'Carnegie Hall']
LatitudeNY  = [40.74837000000008, 40.76746000000003, 40.705790000000036, 40.70765000000006, 40.68969000000004, 40.758290000000045, 40.78300000000007, 40.75407000000007, 40.75648000000007, 40.74967000000004, 40.76494993060773]
LongitudeNY = [-73.98463999999996, -73.97070999999994, -74.00987999999995, -73.99890999999997, -74.04358999999994, -73.97750999999994, -73.95899999999995, -73.97637999999995, -73.98617999999993, -73.96916999999996, -73.9804299522477]

dfNY = pd.DataFrame(columns = {'Location','Latitude','Longitude'})
dfNY.columns = ['Location','Latitude','Longitude']
dfNY['Location']  = LocationNY
dfNY['Latitude']  = LatitudeNY
dfNY['Longitude'] = LongitudeNY

dfNY


# Create Folium map with:
# *  **CircleMarker** with world locations
# *  **MarkerCluster** with New York landmarks 
# 
# Zoom in on New York City to see the cluster open up.  Click on the icon for landmark names.

# In[ ]:


map_world_NYC = folium.Map(location=[df.Latitude.mean(), df.Longitude.mean()],
                       tiles = 'openstreetmap', 
                       zoom_start = 1)

#  CIRCLE MARKERS
#------------------------------
for lat, lng, label in zip(df.Latitude, df.Longitude, df.Location):
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        fill=True,
        color='black',
        fill_color='red',
        fill_opacity=0.6
        ).add_to(map_world_NYC)
#------------------------------

    
#  MARKERS CLUSTERS
#------------------------------
marker_cluster = MarkerCluster().add_to(map_world_NYC)
for lat, lng, label in zip(dfNY.Latitude, dfNY.Longitude, dfNY.Location):
    folium.Marker(location=[lat,lng],
            popup = label,
            icon = folium.Icon(color='green')
    ).add_to(marker_cluster)

map_world_NYC.add_child(marker_cluster)
#------------------------------

#  display map
map_world_NYC         


# [top](#top)
# 
# ---
# # 3.  Choropleth Maps <a id="choro"></a>
# 
# **Choropleth maps** provide a visualization of statistical measurements, such as population density or per-capita income, overlaid on a geographic area.  A **geoJSON** file that defines the areas/boundaries of the state, county, or country is required.

# ###  3.1  Create Dataframe<a id="choro_df"></a>
# `suicide rates from 1986 to 2016` dataset will be used to create a dataframe that contains 'country' and 'suicides/100k pop' for the year 2013.
# 
#   *  Some of country names will be updated to match the geoJSON file
#   *  Once the data for 2013 has been created, the 'year' column can be ignored
#   *  Data is available for only a limited number of countries

# In[ ]:


dfs = pd.read_csv('../input/suicide-rates-overview-1985-to-2016/master.csv')  # suicide rates dataset

dfs = dfs[dfs['year'] == 2013]
dfs = dfs[['country','year','suicides/100k pop']].groupby('country').sum()
dfs.reset_index(inplace=True)

#  update names to match names in geoJSON file
dfs.replace({
        'United States':'United States of America',
        'Republic of Korea':'South Korea',
        'Russian Federation':'Russia'},
        inplace=True)

dfs.head()


# ###  3.2  Read geoJSON File<a id="choro_json"></a>
# `world-countries` dataset will be used for the geoJSON file.
# 
# The geoJSON file contains the shape of the country in multiple latitude/logitude entries:
# `{"type":"Feature","properties":{"name":"Austria"},"geometry":{"type":"Polygon","coordinates":[[[16.979667,48.123497],[16.903754,47.714866],...<deleted>`
# 
# NOTE:  Choropleth map will use the object `feature.properties.name` as a key.

# In[ ]:


world_geo = os.path.join('../input/worldcountries', 'world-countries.json')
world_geo


# ###  3.3  Create Choropleth Map<a id="choro_plot"></a>
# Choropleth Map is created for *'suicides/100k pop'* per *'country'*.
# 
# Note the following components in the Choropleth Map:
#   *  **geo_data** - geoJSON file
#   *  **data** - dataframe for suicide rates
#   *  **columns**  - dataframe columns 'country' and 'suicides/100k pop'
#   *  **key_on**  - from geoJSON file: `feature.properties.name`

# In[ ]:


world_choropelth = folium.Map(location=[0, 0], tiles='Mapbox Bright',zoom_start=1)

world_choropelth.choropleth(
    geo_data=world_geo,
    data=dfs,
    columns=['country','suicides/100k pop'],
    key_on='feature.properties.name',
    fill_color='YlOrRd',
    fill_opacity=0.7, 
    line_opacity=0.2,
    legend_name='Suicide rates per 100k Population - 2013')

folium.LayerControl().add_to(world_choropelth)
# display map
world_choropelth


# ###  EXTRA
# Availble **Folium 'tiles':**   
# tiles = 'cartodbdark_matter',
# 'cartodbpositron',
# 'cartodbpositronnolabels',
# 'cartodbpositrononlylabels',
# 'cloudmade',
# 'mapbox',
# 'mapboxbright',
# 'mapboxcontrolroom',
# 'openstreetmap',
# 'stamenterrain',
# 'stamentoner',
# 'stamentonerbackground',
# 'stamentonerlabels',
# 'stamenwatercolor'.
# 
# Availble **Folium 'fill_color':**   
# fill_color = default 'blue'
# 'BuGn', 'BuPu', 'GnBu', 'OrRd', 'PuBu', 'PuBuGn', 'PuRd', 'RdPu','YlGn', 'YlGnBu', 'YlOrBr', and 'YlOrRd'.
# 
# ---
# [top](#top)
# 
# ##  END
# Please upvote if you found this helpful :-)
