#!/usr/bin/env python
# coding: utf-8

# This notebook is a part of the solution for DSG: EIE competition. The solution splited into 4 parts. Here is the list of notebook in correct order. The part of solution you are currently reading is highlighted in bold.
# 
# [1. Introduction to the solution of DSG: EIE](https://www.kaggle.com/niyamatalmass/1-introduction-to-the-solution-of-dsg-eie)
# 
# [2. Sub-region and State wise E.F and Evaluation](https://www.kaggle.com/niyamatalmass/2-sub-region-and-state-wise-e-f-and-evaluation)
# 
# [**3. Individual Power plant E.F and Evaluation**](https://www.kaggle.com/niyamatalmass/3-individual-power-plant-e-f-and-evaluation)
# 
# [4. Final thoughts, recommendation](https://www.kaggle.com/niyamatalmass/4-final-thoughts-recommendation)
# ***
# <br/>

# <h1 align="center"><font color="#5831bc" face="Comic Sans MS">Individual Power plant E.F and Evaluation</font></h1> 

# # Notebook Overview
# We previously calculate emission and emission factor of Peurto Rico and other states, evaluate with 2018 EPA emissions dataset. We did that excluding pixel that doesn't fall on the power plant area. But we also discuss the trade-off we have to face for evaluation. In this notebook, we will calculate the emission and emission factor for an individual power plant for 2019. After that, we will evaluate with 2019 EPA power plantwise emission data. 
# 

# In[ ]:


import numpy as np
import math 
import pandas as pd
from scipy.ndimage import gaussian_filter
import glob 
import os
import time
from tqdm import tqdm_notebook as tqdm

import geopandas 
import rasterio as rio
import folium
import tifffile as tiff

import ee
from kaggle_secrets import UserSecretsClient
from google.oauth2.credentials import Credentials

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


################
# below code is earth enginie initialization 
# paste it in different cell for initializaiton for yours personal use
################

# Trigger the authentication flow.
# ee.Authenticate()


# !cat ~/.config/earthengine/credentials


user_secret = "earth_engine_2" # Your user secret, defined in the add-on menu of the notebook editor
refresh_token = UserSecretsClient().get_secret(user_secret)
credentials = Credentials(
        None,
        refresh_token=refresh_token,
        token_uri=ee.oauth.TOKEN_URI,
        client_id=ee.oauth.CLIENT_ID,
        client_secret=ee.oauth.CLIENT_SECRET,
        scopes=ee.oauth.SCOPES)
ee.Initialize(credentials=credentials)


# # Calculate emission for all coal power plant
# We previously saw the methodology for calculating emission and emission factor for sub-region and individual states using satellite data. Also, we have tested our methodology against bottom-up emission and found a very good correlation. In this notebook, we are going to calculate the emission and emission factor for an individual power plant. In the above, we described why we should calculate emission and E.F for individual power plant. 
# 
# Methodology: Basically, all the method for calculating emission for an individual power plant is almost identical to our previous method for calculating emission factor for each state. Here we just select each power plant and draw a circle with a relative radius. The relative radius is calculated using the same method. After that, we excluded all the pixels that don't fall into our circle. And finally, reduce that AOI(circle) into total no2 using distance weighted approach. 
# 
# 1. [Importing power plant database, filter them](#power_plant)
# 2. [Create a function for calculating emission for all power plants](#function)
# 3. [Create a batch of running for computation limit and calculate](#batch)
# 4. [Calculate total no2 emission for each power plant](#calculate)
# 5. [Process for evaluation](#process)

# In[ ]:


band_viz = {
  'min': 0,
  'max': 0.0002,
  'palette': ['black', 'blue', 'purple', 'cyan', 'green', 'yellow', 'red']
}

import folium

# def add_ee_layer(self, ee_image_object, vis_params, name):
#     # https://github.com/google/earthengine-api/blob/master/python/examples/ipynb/ee-api-colab-setup.ipynb
#     map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
#     folium.raster_layers.TileLayer(
#     tiles = map_id_dict['tile_fetcher'].url_format,
#     attr = 'Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
#     name = name,
#     overlay = True,
#     control = True
#     ).add_to(self)

    
# Define a method for displaying Earth Engine image tiles on a folium map.
def add_ee_layer(self, ee_object, vis_params, name):
    
    try:    
        # display ee.Image()
        if isinstance(ee_object, ee.image.Image):    
            map_id_dict = ee.Image(ee_object).getMapId(vis_params)
            folium.raster_layers.TileLayer(
            tiles = map_id_dict['tile_fetcher'].url_format,
            attr = 'Google Earth Engine',
            name = name,
            overlay = True,
            control = True
            ).add_to(self)
        # display ee.ImageCollection()
        elif isinstance(ee_object, ee.imagecollection.ImageCollection):    
            ee_object_new = ee_object.mosaic()
            map_id_dict = ee.Image(ee_object_new).getMapId(vis_params)
            folium.raster_layers.TileLayer(
            tiles = map_id_dict['tile_fetcher'].url_format,
            attr = 'Google Earth Engine',
            name = name,
            overlay = True,
            control = True
            ).add_to(self)
        # display ee.Geometry()
        elif isinstance(ee_object, ee.geometry.Geometry):    
            folium.GeoJson(
            data = ee_object.getInfo(),
            name = name,
            overlay = True,
            control = True
        ).add_to(self)
        # display ee.FeatureCollection()
        elif isinstance(ee_object, ee.featurecollection.FeatureCollection):  
            ee_object_new = ee.Image().paint(ee_object, 0, 2)
            map_id_dict = ee.Image(ee_object_new).getMapId(vis_params)
            folium.raster_layers.TileLayer(
            tiles = map_id_dict['tile_fetcher'].url_format,
            attr = 'Google Earth Engine',
            name = name,
            overlay = True,
            control = True
        )
        # display ee.FeatureCollection()
        elif isinstance(ee_object, ee.feature.Feature):  
            ee_object_new = ee.Image().paint(ee_object, 0, 2)
            map_id_dict = ee.Image(ee_object_new).getMapId(vis_params)
            folium.raster_layers.TileLayer(
            tiles = map_id_dict['tile_fetcher'].url_format,
            attr = 'Google Earth Engine',
            name = name,
            overlay = True,
            control = True
        ).add_to(self)
    
    except:
        print("Could not display {}".format(name))
    
def plot_ee_data_on_map(dataset,minimum_value,maximum_value,latitude,longitude,zoom):
    # https://github.com/google/earthengine-api/blob/master/python/examples/ipynb/ee-api-colab-setup.ipynb
    folium.Map.add_ee_layer = add_ee_layer
    vis_params = {
      'min': minimum_value,
      'max': maximum_value,
      'palette': ['black', 'blue', 'purple', 'cyan', 'green', 'yellow', 'red']}
    my_map = folium.Map(location=[latitude,longitude], zoom_start=zoom, height=500)
    my_map.add_ee_layer(dataset, vis_params, 'Color')
    my_map.add_child(folium.LayerControl())
    display(my_map)


# <a id='power_plant'></a>
# ### 1. Importing power plant database, filter them
# As like the previous notebook, we first start by importing our power plant database. Also, in here we imported 2019 EPA emission data. This dataset contains about 1400 power plants emission data for 2019. This makes this dataset perfect for us. Because we have full coverage Sentinel-5p satellite image for whole 2019. So if we calculate individual power plant emission from satellite images we can easily evaluate against this dataset, unlike the previous notebook where we evaluate our model data with slightly different dates. 
# 
# After importing power plant data, we filter them for containing power plants that also available in the EPA 2019 data and for computation easy we only select coal and hydropower plant in this notebook but this can be changed with simple code. 

# In[ ]:


df_2019_epa_emission_data = pd.read_csv('../input/2019-epa-nox-emission/epa_no2_emission_2019.csv')
power_plant_feature_collection = ee.FeatureCollection("WRI/GPPD/power_plants")

country_filter = ee.Filter.eq('country', 'USA')
name_filter = ee.Filter.inList('name', df_2019_epa_emission_data['Facility Name'].tolist())
latitude_filter = ee.Filter.inList('latitude', df_2019_epa_emission_data['Latitude (degrees)'].tolist())
longitude_filter = ee.Filter.inList('longitude', df_2019_epa_emission_data['Longitude (degrees)'].tolist())
type_filter = ee.Filter.inList('fuel1', ['Coal', 'Hydro'])

power_plant_feature_collection = power_plant_feature_collection.filter(
    ee.Filter.And(country_filter,type_filter,
                  ee.Filter.Or([name_filter,
                                latitude_filter,
                                longitude_filter])))


# <a id='function'></a>
# ### 2. Create a function for calculating emission for all power plants
# As we are already familiar with the methodology we don't need much explanation. In the previous notebook, we explained the methodology by giving an individual example but in here we just calculate all power plant emission by one chance. For that, we just declare our function. The function is documented very well and explained every single part of the process. 

# In[ ]:


us_ton_conve_mult_factor = (1.10231e-6)

def calc_each_pp_total_no2(feature):
    buffer_amount = ee.Number(feature.get('gwh_2016')).multiply(1.1)
    buffer_point = feature.geometry().buffer(distance= buffer_amount, maxError=200)
    collection_for_pp = ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_NO2')    .select('tropospheric_NO2_column_number_density')    .filterBounds(buffer_point)
    
    start = ee.Date('2019-01-01')
    finish = ee.Date('2020-01-01')

    diff = finish.difference(start, 'day')
    range_date = ee.List.sequence(0, diff.subtract(1)).map(lambda day: start.advance(day,'day'))
    
    def day_mosaics(date, newlist):
        date = ee.Date(date)
        newlist = ee.List(newlist)
        filtered = collection_for_pp.filterDate(date, date.advance(1,'day'))
        image = ee.Image(filtered.mosaic())
        return ee.List(ee.Algorithms.If(filtered.size(), newlist.add(image), newlist))
    
    collection_for_specific_pp = ee.ImageCollection(ee.List(range_date.iterate(day_mosaics, ee.List([]))))
    
    
#     s5p_mosaic_for_each_states = s5p_mosaic_for_each_states.map(lambda image: image.reproject(
#         crs=collection.first().projection().crs().getInfo(),
#         scale = collection.first().projection().nominalScale().getInfo()))
    
    #######
    maxDist = 20000
    distance = power_plant_feature_collection.distance(searchRadius=maxDist, maxError=1000)
    weight = distance.subtract(maxDist).abs().divide(maxDist)
    
    
    #######
    
    
    collection_for_specific_pp = collection_for_specific_pp.map(lambda image: image.multiply(24500000))
    collection_for_specific_pp = collection_for_specific_pp.map(lambda image: image.multiply(46))
    collection_for_specific_pp = collection_for_specific_pp.map(lambda image: image.multiply(us_ton_conve_mult_factor))
    collection_for_specific_pp = collection_for_specific_pp.reduce(ee.Reducer.sum())
    
    collection_for_specific_pp = collection_for_specific_pp.addBands(weight)
    
    
    ### applying gaussian
    gauss_kernel = ee.Kernel.gaussian(
        radius= 3, sigma= 1,
        units= 'pixels')
    collection_for_specific_pp = collection_for_specific_pp.convolve(gauss_kernel)
    
    
    total_no2_yearly_specific_pp = collection_for_specific_pp.reduceRegion(reducer = ee.Reducer.sum().splitWeights(),
                                                  geometry= buffer_point,
                                                  scale= 1113)
    
    return feature.set({'total_no2' : total_no2_yearly_specific_pp.get('sum')})
#     return feature.set({'total_no2' : total_no2_yearly_specific_pp.get('list').get('tropospheric_NO2_column_number_density_sum')})


# <a id='batch'></a>
# ### 3. Create a batch of running for computation limit and calculate
# For avoiding computation limit, we batched our computation in different parts. Though in this notebook, we don't need that but still we keep the functionality in the case in future we need it. 

# In[ ]:


all_pp_code = power_plant_feature_collection.reduceColumns(
    reducer=ee.Reducer.toList(),
    selectors=['gppd_idnr']).getInfo().get('list')

first_batch_pp_code = all_pp_code[:len(all_pp_code)//1]
second_batch_pp_code = all_pp_code[len(all_pp_code)//2:]


first_batch_pp_collection = power_plant_feature_collection.filter(ee.Filter.inList('gppd_idnr', first_batch_pp_code))

second_batch_pp_collection = power_plant_feature_collection.filter(ee.Filter.inList('gppd_idnr', second_batch_pp_code))


# In[ ]:


first_batch_total_no2 = first_batch_pp_collection.map(lambda feature: calc_each_pp_total_no2(feature))
# second_batch_total_no2 = second_batch_pp_collection.map(lambda feature: calc_each_pp_total_no2(feature))


# <a id='calculate'></a>
# ### 4. Calculate total no2 emission for each power plant
# Finally, using the function we created earlier we calculate all power plant emission using a distance weighted sum approach. How the function works is describes thoroughly in function with comment. 

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nfirst_batch_results = first_batch_total_no2.reduceColumns(\n    reducer=ee.Reducer.toList().repeat(5),\n    selectors=['gppd_idnr', 'name', 'latitude', 'longitude', 'total_no2']).getInfo()")


# In[ ]:


# second_batch_results = second_batch_total_no2.reduceColumns(
#     reducer=ee.Reducer.toList().repeat(5),
#     selectors=['gppd_idnr', 'name', 'latitude', 'longitude', 'total_no2']).getInfo()


# <a id='process'></a>
# ### 5. Process for evaluation
# Finally, we have calculated emission for each power plant. But to evaluate, we have to process the output a little bit. First, we convert our function returns to dataframe and then merge with 2019 power plant wise data. 

# In[ ]:


df_first_batch_total_no2 = pd.DataFrame(columns=['gppd_idnr', 'name', 'latitude', 'longitude', 'total_no2'])
# df_second_batch_total_no2 = pd.DataFrame(columns=['gppd_idnr', 'name', 'latitude', 'longitude', 'total_no2'])


# In[ ]:


def convert_feature_collection_results_to_df(results, dataframe):
    dataframe['gppd_idnr'] = pd.Series(results.get('list')[0])
    dataframe['name'] = pd.Series(results.get('list')[1])
    dataframe['latitude'] = pd.Series(results.get('list')[2])
    dataframe['longitude'] = pd.Series(results.get('list')[3])
    dataframe['total_no2'] = pd.Series(results.get('list')[4])
    return dataframe


# In[ ]:


df_first_batch_total_no2 = convert_feature_collection_results_to_df(first_batch_results, df_first_batch_total_no2)
# df_second_batch_total_no2 = convert_feature_collection_results_to_df(second_batch_results, df_second_batch_total_no2)

# df_pp_total_no2 = pd.concat([df_first_batch_total_no2, df_second_batch_total_no2])


# In[ ]:


temp = df_first_batch_total_no2.merge(df_2019_epa_emission_data, left_on=['latitude', 'longitude'],
                               right_on=['Latitude (degrees)', 'Longitude (degrees)'], how='inner')




temp.head()


# # Evaluation
# In this step, we are going to evaluate our model output for individual power plant. In the previous notebook, we evaluated all states emission factor with bottom-up emission using 2018. But for evaluating power plant emission factor we have full power plant wise data available from EPA for 2019. 
# 
# We first see how far our satellite calculated emission from local emission and then we will see some compare with the distribution. 

# In[ ]:


fig = go.Figure()
fig.add_trace(go.Scatter(
    x=temp['total_no2'].values.tolist(),
    y=temp['name'].values.tolist(),
    marker=dict(color="crimson", size=12),
    mode="markers",
    name="Satellite Image(2019)",
))

fig.add_trace(go.Scatter(
    x=temp['2019 NOX Emissions (tons)'].values.tolist(),
    y=temp['name'].values.tolist(),
    marker=dict(color="gold", size=12),
    mode="markers",
    name="Bottom up(2019)",
))

fig.update_layout(title="Satellite emission data vs bottom up emission data for individual power plants",
                  xaxis_title="NO2 amount (tons)",
                  yaxis_title="Power plant" ,autosize=False,
                  width=700,height=1500,yaxis= {'nticks': 100})

fig.show()


# We are seeing very promising results. We are seeing that our model output is very correlated with bottom-up emission which means that our model performs very well. Now we can easily get the emission factor by dividing the emission of each power plant with its generation. 
