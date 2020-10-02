#!/usr/bin/env python
# coding: utf-8

# # Tutorial: Foursquare API Query

# <img src="https://media.giphy.com/media/xUA7aZ52SKoPBz7PBC/giphy.gif">

# ### In this notebook, I will walk you through how to make calls to the Foursquare API for different purposes. We will learn how to construct a URL to send a request to the API to search for a specific type of venues, explore a particular venue, and explore a geographical location. We will use the visualization library, Folium, to visualize the results.

# ### Why Foursquare? 100,000+ developers, used by Apple Maps, Uber, Snapchat, Twitter, etc. for its location-based service's data to improve the accuracy venues on a map.

# > ## Quick rundown:
# * [Setting up a Developer Account](#first-section)
# * [Basic API Call](#second-section)
# * [Example: Analyzing Pizza in NYC](#third-section)
# * [Data Visualization of Example](#fourth-section)

# # Setting up a Developer Account<a class="anchor" id="first-section"></a>

# 1. Go to https://developer.foursquare.com
# 2. Create an account at https://foursquare.com/developers/signup
# 3. Sign up using your name, email, and password
# 4. Confirm your email
# 5. Create an App in the developer page, enter a website for the App (does not have to be a real App or website)
# 6. Enable Places API
# 7. Click your newly created App for CLIENT ID and CLIENT SECRET (you will use this in the API query)
# 
# For the basic account tier of Sandbox, you get: 950 Regular Calls/Day (calls for general information), 50 Premium Calls/Day (calls about a specific venue), 1 Photo per Venue, 1 Tip per Venue.

# # Basic API Call<a class="anchor" id="second-section"></a>

# ## Importing libraries used in this notebook.

# In[ ]:


import requests # library to handle requests
import pandas as pd # library for data analsysis
import numpy as np # library to handle data in a vectorized manner
import random # library for random number generation

#uncomment next line if need to install latest version of geopy
#!conda install -c conda-forge geopy --yes 
from geopy.geocoders import Nominatim # module to convert an address into latitude and longitude values

# libraries for displaying images
from IPython.display import Image 
from IPython.core.display import HTML 
    
# tranforming json file into a pandas dataframe library
from pandas import json_normalize

#uncomment next line if need to install latest version of folium
#!conda install -c conda-forge folium=0.5.0 --yes
import folium # plotting library

print('Folium installed')
print('Libraries imported.')


# ## Define Foursquare Credentials and Version

# In[ ]:


CLIENT_ID = 'your-client-ID' # enter your Foursquare ID here!
CLIENT_SECRET = 'your-client-secret' # enter your Foursquare Secret here!

VERSION = '20180604' # what version of Foursquare you want to use
LIMIT = 20 # max limit is 50 
print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# In[ ]:


CLIENT_ID = '0IBRWROBN4BHTVCXD5J43BB3JQARQ4V1DRETPONPPMVY205B'
CLIENT_SECRET = 'KTMQCKPDDQFYN41SU0R1OG3DTW4DHHZARAAKNLSJF0DXJSYH' 


# # Example: Analyzing Pizza in NYC<a class="anchor" id="third-section"></a>

# ## Let's assume someone just got off Grand Central Terminal, NYC. We can convert this address to latitude and longitude coordinates.

# In[ ]:


# Grand Central Terminal Address
address = '89 E 42nd St, New York, NY 10017'

geolocator = Nominatim(user_agent="foursquare_agent")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print("The latitude and longitude coordinates are:")
print(latitude, longitude)


# ## They want to find some pizza to eat.

# In[ ]:


search_query = 'Pizza'
radius = 500 #Radius of search in meters

url = 'https://api.foursquare.com/v2/venues/search?client_id={}&client_secret={}&ll={},{}&v={}&query={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET, latitude, longitude, VERSION, search_query, radius, LIMIT)
results = requests.get(url).json()


# ## Get relevant part of JSON and transform it into a *pandas* dataframe

# In[ ]:


# assign relevant part of JSON to venues
venues = results['response']['venues']

# tranform venues into a pandas dataframe
dataframe = json_normalize(venues)
dataframe.head()


# ## Define information of interest and filter the dataframe

# In[ ]:


# keep only columns that include venue name, and anything that is associated with location
filtered_columns = ['name', 'categories'] + [col for col in dataframe.columns if col.startswith('location.')] + ['id']
dataframe_filtered = dataframe.loc[:, filtered_columns]

# function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']

# filter the category for each row
dataframe_filtered['categories'] = dataframe_filtered.apply(get_category_type, axis=1)

# clean column names by keeping only last term
dataframe_filtered.columns = [column.split('.')[-1] for column in dataframe_filtered.columns]

dataframe_filtered


# ## What's the rating of the 3 closest pizza place?

# In[ ]:


venue_id = '4c7d96bbd65437043defc0a2' # ID of closest pizza joint
url = 'https://api.foursquare.com/v2/venues/{}?client_id={}&client_secret={}&v={}'.format(venue_id, CLIENT_ID, CLIENT_SECRET, VERSION)

result = requests.get(url).json()
result['response']['venue']
print(result['response']['venue']['name'])
try:
    print(result['response']['venue']['rating'])
except:
    print('This venue has not been rated yet.')


# In[ ]:


venue_id = '4a8c31aef964a520410d20e3' # ID of second closest
url = 'https://api.foursquare.com/v2/venues/{}?client_id={}&client_secret={}&v={}'.format(venue_id, CLIENT_ID, CLIENT_SECRET, VERSION)

result = requests.get(url).json()
print(result['response']['venue']['name'])
try:
    print(result['response']['venue']['rating'])
except:
    print('This venue has not been rated yet.')


# In[ ]:


venue_id = '4d012c08ba1da1cd3cb68c28' # ID of third closest
url = 'https://api.foursquare.com/v2/venues/{}?client_id={}&client_secret={}&v={}'.format(venue_id, CLIENT_ID, CLIENT_SECRET, VERSION)

result = requests.get(url).json()
print(result['response']['venue']['name'])
try:
    print(result['response']['venue']['rating'])
except:
    print('This venue has not been rated yet.')


# # Data Visualization of Example<a class="anchor" id="fourth-section"></a>

# In[ ]:


venues_map = folium.Map(location=[latitude, longitude], zoom_start=16) # generate map centred around the Grand Central Terminal

# add a red circle marker to represent Grand Central Terminal
folium.features.CircleMarker(
    [latitude, longitude],
    radius=10,
    color='red',
    popup='Grand Central Terminal',
    fill = True,
    fill_color = 'red',
    fill_opacity = 0.6
).add_to(venues_map)

# add the pizza joints as blue circle markers
for lat, lng, label in zip(dataframe_filtered.lat, dataframe_filtered.lng, dataframe_filtered.categories):
    folium.features.CircleMarker(
        [lat, lng],
        radius=5,
        color='blue',
        popup=label,
        fill = True,
        fill_color='blue',
        fill_opacity=0.6
    ).add_to(venues_map)

# display map
venues_map


# In[ ]:




