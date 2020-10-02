#!/usr/bin/env python
# coding: utf-8

# # Visualizing Median Home Values by Mapping Choropleths
# 
# This notebook will demonstrate how to use the [IdealSpot API](https://developer.idealspot.com/) to query Median Home Values by CENSUS County Tract and create a Choropleth Map.  A Choropleth Map displays divided geographical areas or regions that are coloured, shaded or patterned in relation to a data variable. This demo also uses [census](https://github.com/datamade/census), a simple API wrapper for US CENSUS data.
# 
# For our example, we will examine CENSUS County Tracts in Travis County, Texas, where Austin, TX is located.
# 
# ## Requirements
# 
# To run this notebook, you'll need to get two API keys.
# 
# 1. US CENSUS API Key. You can request one [here](https://api.census.gov/data/key_signup.html).
# 2. IdealSpot API Key.  You can get a trial key [here](https://rapidapi.com/idealspot-inc-idealspot-inc-default/api/idealspot-geodata)
# 

# In[ ]:


get_ipython().run_cell_magic('capture', '', '!pip install census\n!pip install us')


# In[ ]:


# standard libraries
from datetime import datetime
import json
import os
import requests

# 3rd party librares
import branca.colormap as cm
from census import Census
from IPython.display import display
import folium
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
from us import states

# local
from kaggle_secrets import UserSecretsClient


# In[ ]:


# secrets
user_secrets = UserSecretsClient()
CENSUS_KEY = user_secrets.get_secret("CENSUS_KEY")
RAPID_API_KEY = user_secrets.get_secret("RAPID_API_KEY")

# RapidAPI Request Headers (kevincal unlimited key)
API_URL = "https://idealspot-geodata.p.rapidapi.com"

API_REQUEST_HEADERS = {
    'x-rapidapi-host': "idealspot-geodata.p.rapidapi.com",
    'x-rapidapi-key': RAPID_API_KEY
    }


# ## Country Geographic Data Lookup
# 
# Since we want to query all CENSUS Tracts inside Travis County, TX, we need to get the FIPS number for Travis County. The Federal Information Processing Standard Publication 6-4 (FIPS 6-4) was a five-digit Federal Information Processing Standards code which uniquely identified counties and county equivalents in the United States, certain U.S. possessions, and certain freely associated states. [more](https://en.wikipedia.org/wiki/FIPS_county_code)
# 
# We use IdealSpot's [Geometry lookup API EndPoint](https://rapidapi.com/idealspot-inc-idealspot-inc-default/api/idealspot-geodata?endpoint=apiendpoint_747b0075-daa3-4d5c-8ebb-7328bbea8cc3)
# 
# 
# 

# In[ ]:


# IdealSpot
url = API_URL + "/api/v1/geometries/geometry"

# Build `Location` Object to Query API
location = {
    "type":"region",
    "regiontype": "county",
    "region_id": "TravisCountyTX"
}

# Fetch Geometry Record from IdealSpot API
params = { "location": json.dumps(location) }
r = requests.get(url=url, params=params, headers=API_REQUEST_HEADERS)
data = r.json().get('data')
display(data)


# In[ ]:


# looking at the `Feature Collection` above, get FIPS for Travis County, TX

# Get County FIPS code
travis_fips = data['features'][0]['properties']['_properties']['COUNTYFP']

# using FIPS, get the list of CENSUS County Tracts 
c = Census(CENSUS_KEY)
county_tracts = c.sf1.state_county_tract('NAME', states.TX.fips, travis_fips, Census.ALL)

# display
print("%s county tracts in Travis County, TX" % len(county_tracts))
display(county_tracts[:2])


# ## Check Home Value API Request
# 
# IdealSpot has the ability to fetch request/response structure metadata for a given Insight. This provides you the periods of data available as well as any other parameters you may want to query the Insight by. Example Insights include population and market data such as: age, daytime population, avg. home. [Describe Insight API End Point](https://rapidapi.com/idealspot-inc-idealspot-inc-default/api/idealspot-geodata?endpoint=apiendpoint_2d3cd858-6264-40c7-938a-890eded9511d)
# 

# In[ ]:


# get home-value for East Austin
url = "https://idealspot-geodata.p.rapidapi.com/api/v1/data/insights/home-value"

# fetch API endpoint and display results
params = {
    "periods": "true",
    "parameter_options": "true",
    "parameters": "true"
    }

r = requests.get(url=url, params=params, headers=API_REQUEST_HEADERS)
display(r.json())


# ## Iterate and Query for each CENSUS County Tracts
# 
# From the above metadata we know we want to query `version=v2` for `home-value`.
# 

# In[ ]:


# tip: keep a list of processed id's so no need to requery
processed_region_ids = []

# list of county tract data
county_tract_data = []


# In[ ]:


# iterate through county tracts to fetch geo and insight data
for ct in county_tracts:
    
    # debug 
    # print("Processing %s" % ct)
    
    # get tract FIPS
    name = ct.get('NAME')
    state_fips = ct.get('state')
    county_fips = ct.get('county')
    tract_fips = ct.get('tract')
    
    # build region_id
    region_id = "%s%s%s" % (state_fips, county_fips, tract_fips)
    
    # get duplicate
    if region_id in processed_region_ids:
        print("Duplicate. Skipping")
        continue
    else:
        processed_region_ids.append(region_id)
    
    # build location parameter for API Query
    location = {
        "type":"region", 
        "regiontype": "tract", 
        "region_id": region_id
        }
    
    # API params, using 
    params = {
        "version": "v2",
        "location": json.dumps(location)
    }
    
    # build Insight API URL
    url = API_URL + "/api/v1/data/insights/home-value/query"
    r = requests.get(url=url, params=params, headers=API_REQUEST_HEADERS)
    insight_raw_data = r.json()
    insight_data = insight_raw_data.get('data')[0].get('data')
    median_home_value = None
    for (l, v) in insight_data:
        if l == 'Median Home Value':
            median_home_value = v
            
    # get Geometry API URL
    url = API_URL + "/api/v1/geometries/geometry"
    r = requests.get(url=url, params=params, headers=API_REQUEST_HEADERS)    
    geometry_raw_data = r.json()
    
    # get the polygon coordinates
    geometry = geometry_raw_data.get('data', {})        .get('features', [])[0].get('geometry')
    coordinates = geometry['coordinates'][0]
    poly = Polygon(coordinates)
    
    # build list
    county_tract_data.append({
        'region_name': name,
        'region_id': int(region_id),
        'Median Home Value': median_home_value,
        'geometry': poly
    })


# ## GeoPandas
# 
# [GeoPandas](https://github.com/geopandas/geopandas) is a project to add support for geographic data to pandas objects. It currently implements `GeoSeries` and `GeoDataFrame` types which are subclasses of `pandas.Series` and `pandas.DataFrame` respectively. GeoPandas objects can act on shapely geometry objects and perform geometric operations.
# 
# We load the `country_tracks_data` into a `GeoDataFrame` so we can easily process and plot this data.

# In[ ]:


# create GeoDataFrame
county_tracts_gdf = gpd.GeoDataFrame(county_tract_data, crs="EPSG:4326")
county_tracts_gdf.set_index('region_id')
county_tracts_gdf.head()


# ## Folium
# 
# [folium](https://python-visualization.github.io/folium/) makes it easy to visualize data that's been manipulated in Python on an interactive leaflet map. It enables both the binding of data to a map for choropleth visualizations as well as passing rich vector/raster/HTML visualizations as markers on the map.
# 

# In[ ]:


# center map based on features
lng_map = county_tracts_gdf.centroid.x.mean()
lat_map = county_tracts_gdf.centroid.y.mean()

# create folium map
map = folium.Map(
    location=[lat_map, lng_map],
    zoom_start=11,
    tiles=None)

# set tilelayer manually for more control
tile_layer = folium.TileLayer('CartoDB positron', name="Light Map", control=False).add_to(map)

# build color scale
threshold_scale = (county_tracts_gdf['Median Home Value']
                       .quantile((0, 0.1, 0.75, 0.9, 0.98, 1))).tolist()

colormap = cm.linear.YlGnBu_09.to_step(
    data=county_tracts_gdf['Median Home Value'], 
    method='quant', 
    quantiles=[0, 0.1, 0.75, 0.9, 0.98,1 ])

# build choropleth
folium.Choropleth(
     geo_data=county_tracts_gdf,
     name='Median Home Value',
     data=county_tracts_gdf,
     columns=['region_id', 'Median Home Value'],
     key_on="feature.properties.region_id",
     fill_color='YlGnBu',
     threshold_scale=threshold_scale,
     fill_opacity=0.6,
     line_opacity=0.2,
     legend_name='Median Home Value',
     smooth_factor=0
    ).add_to(map)

# display the static map
display(map)


# ## Interactive Tooltip
# 
# We're going to add some interactivity.  
# 1. Hover / Tooltip
# 2. Layer Controller (to show/hide the Choropleth)

# In[ ]:


# build interactive tooltip
style_function = lambda x: {
    'weight': 0.1, 
    'color': 'black',
    'fillColor': colormap(x['properties']['Median Home Value']), 
    'fillOpacity': 0.01
    }

highlight_function = lambda x: {
    'fillColor': '#000000', 
    'color':'#000000', 
    'fillOpacity': 0.2, 
    'weight': 0.1
    }

tooltip_overlay=folium.features.GeoJson(
        county_tracts_gdf,
        style_function=style_function,
        control=False,
        highlight_function=highlight_function,
        tooltip=folium.features.GeoJsonTooltip(
            fields=['region_name', 'Median Home Value'],
            aliases=['County Tract', 'Median Home Value ($)'],
            style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;"),
            sticky=True,
            localize=True
        )
    )
map.add_child(tooltip_overlay)
map.keep_in_front(tooltip_overlay)

# add layer control
folium.LayerControl().add_to(map)

# display map
display(map)


# # That's a Wrap
# 
# So in this tutorial you learned how to use the [IdealSpot API](http://developer.idealspot.com) to query for CENSUS County Tract median home values and plot them on an interactive map. Explore the other datasets and geographic areas available in our API to see what more interesting data you can plot!

# In[ ]:




