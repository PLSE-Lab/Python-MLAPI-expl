#!/usr/bin/env python
# coding: utf-8

# # <font color='blue'>Kiva local information of loans</font>

# Kiva has mentioned in the purpose of the project that localisation is very important for them to understand the local needs. Indeed, looking into the local areas, you can see more details about the needs. A previous kernel has included information on the approximate longitude and latitude of the areas of the loan. This notebook, extends on this information by looking at some important basic facilities in those areas. In this way, we can enhance the information on a local level.
# The information includes the following in a radius of 10KM in the latitude/longitude provided:
# * \# universities and colleges
# * \# hospitals
# * \# airports
# * \# cities
# * \# hotels

# ## Import libraries
# 

# In[ ]:


get_ipython().system(' pip install --upgrade pip ')
get_ipython().system(' pip install googlemaps')
get_ipython().system(' pip install geopy')
get_ipython().system('pip install python-google-places')


# In[ ]:


import googlemaps
import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
import seaborn as sns
from geopy.distance import vincenty

import requests
import logging
import time


import re

get_ipython().run_line_magic('matplotlib', 'inline')


# https://console.cloud.google.com/apis/dashboard
# 
# Enabled APIs
# * Places
# * Geocoding
# 
# goglemaps library documentation:
# https://googlemaps.github.io/google-maps-services-python/docs/

# In[ ]:


key = 'AIzaSyCOdepFwT2KhXt0QomoLS-MZUve2UAp3Q8'
gmaps = googlemaps.Client(key=key)


#Generic parameters to be set
search_radius = 10000
output_folder = './google_maps_data'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

keywords = ['university or college','hotel','airport','city','hospitals'] # add more places later


# ## Function with all the elements to be googled

# In[3]:


def get_google_info(kiva_regions):
    '''
    given the name of the region, after some pre-processing to clean the data, each location is 
    searched on google maps and different elements are searched. 
    '''
    region_info = dict()
    perc_modulo = int(kiva_regions.shape[0]/10.)
    print("the number of regions to be analysed is " + str(kiva_regions.shape[0]))
    for index in range(kiva_regions.shape[0]):
        
        region_number = kiva_regions.iloc[index]['LocationName']
        if index % perc_modulo == 0:
            print('{}%'.format(float(index)/kiva_regions.shape[0]))
        if region_number in region_info:
            continue
        output_file_name = os.path.join(output_folder, '{}_store_data.json'.format(region_number))
        
        region_info[region_number] = dict()
        data = {
            'region_number': region_number,
            'nearby': dict(),
        }

        # Get the region location

        region_location = gmaps.geocode(region_number)

        data['geocode'] = region_location
        if len(region_location) > 0:
            region_geolocation = region_location[0]['geometry']['location']
            for keyword in keywords:
                # Search for nearby places
                nearby = gmaps.places_nearby(
                    keyword=keyword,
                    location=region_geolocation,
                    radius=search_radius)
                region_info[region_number][keyword] = len(nearby['results'])
                region_info[region_number]['area_type'] =  str(data['geocode'][0]['address_components'][0]['types'][0])
                data['nearby'][keyword] = nearby
        else:
            region_geolocation = None
            for keyword in keywords:
                region_info[region_number][keyword] = 0

        with open(output_file_name, 'w') as fout:
            json.dump(data, fout)
            
    print("all the locations have been searched!")
    
    return region_info
   


# In[ ]:


def get_google_info_given_coords(coords):
    '''
    given the name of the region, after some pre-processing to clean the data, each location is 
    searched on google maps and different elements are searched. 
    '''
    global region_info
    region_info = dict()
    perc_modulo = int(coords.shape[0]/10.)
    print("the number of regions to be analysed is " + str(coords.shape[0]))
    for index in range(coords.shape[0]):
        
        region_number = (coords.iloc[index]['latitude'],coords.iloc[index]['longitude'])
        if index % perc_modulo == 0:
            print('{}%'.format(float(index)/coords.shape[0]))
        if region_number in region_info:
            continue
        output_file_name = os.path.join(output_folder, '{}_store_data.json'.format(region_number))
        
        region_info[region_number] = dict()
        data = {
            'region_number': region_number,
            'nearby': dict(),}

        region_geolocation = {u'lat': coords_unique.iloc[index]['latitude'], u'lng':coords_unique.iloc[index]['longitude']}
        for keyword in keywords:
            # Search for nearby places
            nearby = gmaps.places_nearby(
                    keyword=keyword,
                    location=region_geolocation,
                    radius=search_radius)
            region_info[region_number][keyword] = len(nearby['results'])

    print("all the locations have been searched!")
    
    return region_info
   


# In[ ]:


def table_google_features_v2(region_info):
    X = pd.DataFrame.from_dict(region_info,orient='index')
    X['coords'] = X.index
    X = X.reset_index(drop=True)
    
    return X


# ## Read in data

# In[ ]:


regions = pd.read_csv('kiva_mpi_region_locations.csv')
regions = regions[['LocationName','geo']].drop_duplicates()

kiva_regions = pd.read_csv('kiva_loans.csv')
kiva_regions = kiva_regions[['region','country']].dropna(axis=0).drop_duplicates()
#clean names
kiva_regions['region_clean'] = kiva_regions['region'].apply(lambda x:re.sub('\W+',' ',x))
kiva_regions['LocationName'] = kiva_regions['region_clean'] + ', ' + kiva_regions['country']

google_combine = pd.read_csv('loan_coords_additionalinfo.csv')
google_combine = google_combine[['university or college','airport','hotel','hospitals','city','coords']]
print("at the moment there is information for " + str(google_combine.shape[0]) + " unique regions...")

kiva_loans_coords = pd.read_csv('loan_coords.csv')
coords_unique = kiva_loans_coords[['latitude','longitude']].drop_duplicates()
print("there are " + str(coords_unique.shape[0]) + " coordinate locations in the kiva loan dataset...")


# ## Run function and collect information from google API

# In[5]:


# when you have the coordinates of the region
out = get_google_info_given_coords(coords_unique[200:400]) # there is a daily quota for free 
gdata = table_google_features_v2(region_info)


# In[ ]:


file_save = pd.concat([google_combine,gdata])
file_save.to_csv('loan_coords_additionalinfo.csv',index=False)


# 
