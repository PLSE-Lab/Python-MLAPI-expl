#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This report analyzes user restaurant reviews based on a geographic location in an effort to provide possible lunch options for future planning. We will be leveraging Foursquare API in order to collect and analyze data for specific type of venues, users, geographical location and get trending venues around specific location. 
# 
# The problem in question that we are trying to solve is "What should we eat for lunch?" & "If time permits, we would like to go some other popular spots. What other places we can go to after lunch?"

# # Contents
# 
# 1- Introduction
# 
# 2- About the Data Set
# 
# 3- Data Collection and Understanding
# 
# 4- Data Exploration 
# 
# 5- Conclusion

# # About the Data Set
# 
# Foursquare allows developers to be able to create an app and use their service to connect the data collected and available by Foursquare. In this case, we will be using their free foresquare developer account option with certain limitations however it is possible to become a premium member and get full access to the Foresquare API. Data variables are as explained below;
# 
# #### Name: Name of the Venue
# #### Categories: Category of the Venue
# #### Address: Address of the Venue
# #### CC: Country Code
# #### City: City of the Venue
# #### Country: Country of the Venue
# #### Cross Street: Cross street of the Venue
# #### distance: Distance from the provided location
# #### formattedAddress: Address of the venue
# #### labeledLatLngs	: Label of the venue
# #### lat: Latitude of the venue
# #### lng: longtitude of the venue
# #### postalCode: Zip code of the venue
# #### state: State of the venu
# #### id: unique id for the venue

# In[ ]:


# import neccessary libraries
get_ipython().system('conda install -c conda-forge geopy --yes ')
from geopy.geocoders import Nominatim # module to convert an address into latitude and longitude values
import requests # library to handle requests
import pandas as pd # library for data analsysis
import numpy as np # library to handle data in a vectorized manner
import random # library for random number generation

# libraries for displaying images
from IPython.display import Image 
from IPython.core.display import HTML 
    
# tranforming json file into a pandas dataframe library
from pandas.io.json import json_normalize

get_ipython().system('conda install -c conda-forge folium=0.5.0 --yes')
import folium # plotting library

print('Folium installed')
print('Libraries imported.')


# # Data Collection and Understanding
# 
# In order to use the Foursquare API and collect and analyze the available data, we need to create a Foursqaure Developer Account, and create an app. 

# In[ ]:


# Define Foursquare Credentials and Version

CLIENT_ID = 'SLDDVV4ZMUHC2TGZ113CKT1XF3MJZRBZNRL4QSYVDRGLW3XJ' # your Foursquare ID
CLIENT_SECRET = 'VDOU35CORKHKU3PPHIMZDFZDLYSHH4HRZAQGVZGYIENPRHNA' # your Foursquare Secret
VERSION = '20180604'
LIMIT = 30
print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# We are currently at City University of New York which is located at 365 5th Ave, New York, NY 10016. We would want to have lunch close by.

# In[ ]:


address = '365 5th Ave, New York, NY 10016'

geolocator = Nominatim()
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print(latitude, longitude)


# The way the Foresqaure API is structured, we can search for a specific venue category data via JSON as below structure
# 
# > `https://api.foursquare.com/v2/venues/`**search**`?client_id=`**CLIENT_ID**`&client_secret=`**CLIENT_SECRET**`&ll=`**LATITUDE**`,`**LONGITUDE**`&v=`**VERSION**`&query=`**QUERY**`&radius=`**RADIUS**`&limit=`**LIMIT**

# We want to eat Japanese food for lunch so we can add that as part of our query.

# In[ ]:


search_query = 'Japanese'
radius = 500
print(search_query + ' .... OK!')


# In[ ]:


# create the api url
url = 'https://api.foursquare.com/v2/venues/search?client_id={}&client_secret={}&ll={},{}&v={}&query={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET, latitude, longitude, VERSION, search_query, radius, LIMIT)
url


# In[ ]:


# get the data result in json
results = requests.get(url).json()
results


# In[ ]:


# assign relevant part of JSON to venues
venues = results['response']['venues']

# tranform venues into a dataframe
dataframe = json_normalize(venues)
dataframe.head()


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


# In[ ]:


dataframe_filtered.info()


# In[ ]:


dataframe_filtered.describe()


# Below are the list of available restaurants
# 
# - Ten Sushi Japanese Restaurant
# - Umi Japanese Restaurant
# - Waka Japanese Dining
# - Hapon Japanese Restaurant
# - Sariku

# # Data Exploration and Visualization

# In[ ]:


dataframe_filtered.name


# In[ ]:


venues_map = folium.Map(location=[latitude, longitude], zoom_start=13) # generate map centred around the CUNY

# add a red circle marker to represent the CUNY
folium.features.CircleMarker(
    [latitude, longitude],
    radius=10,
    color='red',
    popup='CUNY',
    fill = True,
    fill_color = 'red',
    fill_opacity = 0.6
).add_to(venues_map)

# add the Japanese restaurants as blue circle markers
for lat, lng in zip(dataframe_filtered.lat, dataframe_filtered.lng):
    folium.features.CircleMarker(
        [lat, lng],
        radius=5,
        color='blue',
        fill = True,
        fill_color='blue',
        fill_opacity=0.6
    ).add_to(venues_map)

# display map
venues_map


# Based on the distance variable, closest venue is Ten Sushi Place. Let's explore that venue further.

# In order to get more detailed data from a specific venue, we can use the Foresquare API with below format to get all the data in JSON format.
# > `https://api.foursquare.com/v2/venues/`**VENUE_ID**`?client_id=`**CLIENT_ID**`&client_secret=`**CLIENT_SECRET**`&v=`**VERSION**

# In[ ]:


venue_id = '4e4e4ad5bd4101d0d7a7002d' # ID of Ten Sushi Restaurant
url = 'https://api.foursquare.com/v2/venues/{}?client_id={}&client_secret={}&v={}'.format(venue_id, CLIENT_ID, CLIENT_SECRET, VERSION)
url


# In[ ]:


result = requests.get(url).json()
print(result['response']['venue'].keys())
result['response']['venue']


# We can get more specific and get the rating of Ten Sushi Restaurant.

# In[ ]:


try:
    print(result['response']['venue']['rating'])
except:
    print('This venue has not been rated yet.')


# It looks like Ten Sushi Restaurant has no ratings. Let's try a different Japanese Restaurant. Sariku Japanese Restaurant is the second closest.

# In[ ]:


venue_id = '4c44df40429a0f47c660491e' # ID of Sariku Japanese Restaurant
url = 'https://api.foursquare.com/v2/venues/{}?client_id={}&client_secret={}&v={}'.format(venue_id, CLIENT_ID, CLIENT_SECRET, VERSION)

result = requests.get(url).json()
try:
    print(result['response']['venue']['rating'])
except:
    print('This venue has not been rated yet.')


# Unfortunately Sariku Restaurant does not have any rating as well. But we can further look at details for this specific venue. We can review the tips variable via the data collected from Foursquare API.

# In[ ]:


# get the number of tips
result['response']['venue']['tips']['count']


# We need the below format in order to get the tips detail.
# > `https://api.foursquare.com/v2/venues/`**VENUE_ID**`/tips?client_id=`**CLIENT_ID**`&client_secret=`**CLIENT_SECRET**`&v=`**VERSION**`&limit=`**LIMIT**

# In[ ]:


## Sariku Tips - create the url
limit = 15 # set limit to be greater than or equal to the total number of tips
url = 'https://api.foursquare.com/v2/venues/{}/tips?client_id={}&client_secret={}&v={}&limit={}'.format(venue_id, CLIENT_ID, CLIENT_SECRET, VERSION, limit)

# get the url in json
results = requests.get(url).json()
results


# In[ ]:


tips = results['response']['tips']['items']

tip = results['response']['tips']['items'][0]
tip.keys()


# In[ ]:


pd.set_option('display.max_colwidth', -1)

tips_df = json_normalize(tips) # json normalize tips

# columns to keep
filtered_columns = ['text', 'agreeCount', 'disagreeCount', 'id', 'user.firstName', 'user.lastName', 'user.gender', 'user.id']
tips_filtered = tips_df.loc[:, filtered_columns]

# display tips
tips_filtered


# As you can see we can get full information of the user left feedback for Umi Japanese Restaurant. Please the reason we only received 3 of the restaurants tip instead of more is because we are using free developer account. Premium members can get more information.

# We can further research the foresquare user and see if he is a fan of Japanese food or if it was a one of tip he left as a comment for the Umi Restaurant.

# In order to do this, we need to set up our api url structure like this
# > `https://api.foursquare.com/v2/users/`**USER_ID**`?client_id=`**CLIENT_ID**`&client_secret=`**CLIENT_SECRET**`&v=`**VERSION**

# In[ ]:


user_id = '8785316' # user ID of Andrew Buck

# create the url
url = 'https://api.foursquare.com/v2/users/{}?client_id={}&client_secret={}&v={}'.format(user_id, CLIENT_ID, CLIENT_SECRET, VERSION) # define URL

# send GET request
results = requests.get(url).json()
user_data = results['response']['user']

# display features associated with user
user_data.keys()


# In[ ]:


# full information on Andrew Buck
print('First Name: ' + user_data['firstName'])
print('Last Name: ' + user_data['lastName'])
print('Home City: ' + user_data['homeCity'])


# In[ ]:


# figuring out how many tips have been submitted.
user_data['tips']


# In[ ]:


# define tips URL
url = 'https://api.foursquare.com/v2/users/{}/tips?client_id={}&client_secret={}&v={}&limit={}'.format(user_id, CLIENT_ID, CLIENT_SECRET, VERSION, limit)

# send GET request and get user's tips
results = requests.get(url).json()
tips = results['response']['tips']['items']

# format column width
pd.set_option('display.max_colwidth', -1)

tips_df = json_normalize(tips)

# filter columns
filtered_columns = ['text', 'agreeCount', 'disagreeCount', 'id']
tips_filtered = tips_df.loc[:, filtered_columns]

# display user's tips
tips_filtered


# We can also look at the Foresquare User's connections and friends. 

# In[ ]:


user_friends = json_normalize(user_data['friends']['groups'][0]['items'])
user_friends


# In[ ]:


# getting users profile image
user_data


# In[ ]:


# we can pull the image 
Image(url='https://fastly.4sqi.net/img/user/300x300/KRKQK1XSOZZSATJB.jpg')


# To answer the second part of our question on where else we can go once we are done with lunch, we can explore the location. In order to do that we need to have the api url structure as below
# > `https://api.foursquare.com/v2/venues/`**explore**`?client_id=`**CLIENT_ID**`&client_secret=`**CLIENT_SECRET**`&ll=`**LATITUDE**`,`**LONGITUDE**`&v=`**VERSION**`&limit=`**LIMIT**

# In[ ]:


# umi restaurant latitude and longitude are as below
latitude=40.744587
longitude=-73.981918


# In[ ]:


# define the url
url = 'https://api.foursquare.com/v2/venues/explore?client_id={}&client_secret={}&ll={},{}&v={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET, latitude, longitude, VERSION, radius, LIMIT)
url

# get the url in json
results = requests.get(url).json()
'There are {} around Umi restaurant.'.format(len(results['response']['groups'][0]['items']))


# In[ ]:


# get the relevant part of the JSON
items = results['response']['groups'][0]['items']
items[0]


# In[ ]:


# create a clean dataframe
dataframe = json_normalize(items) # flatten JSON

# filter columns
filtered_columns = ['venue.name', 'venue.categories'] + [col for col in dataframe.columns if col.startswith('venue.location.')] + ['venue.id']
dataframe_filtered = dataframe.loc[:, filtered_columns]

# filter the category for each row
dataframe_filtered['venue.categories'] = dataframe_filtered.apply(get_category_type, axis=1)

# clean columns
dataframe_filtered.columns = [col.split('.')[-1] for col in dataframe_filtered.columns]

dataframe_filtered.head(10)


# In[ ]:


# visualize the items on the map
venues_map = folium.Map(location=[latitude, longitude], zoom_start=15) # generate map centred around Umi Restaurant


# add Umi Restaurant as a red circle mark
folium.features.CircleMarker(
    [latitude, longitude],
    radius=10,
    popup='Umi Restaurant',
    fill=True,
    color='red',
    fill_color='red',
    fill_opacity=0.6
    ).add_to(venues_map)


# add popular spots to the map as blue circle markers
for lat, lng, label in zip(dataframe_filtered.lat, dataframe_filtered.lng, dataframe_filtered.categories):
    folium.features.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        fill=True,
        color='blue',
        fill_color='blue',
        fill_opacity=0.6
        ).add_to(venues_map)

# display map
venues_map


# Now we found a Japanese Restaurant, places to go after lunch, we can further analyze the other venues to see which ones are trending within FoureSquare users. We need to structure our url as follows;
# > `https://api.foursquare.com/v2/venues/`**trending**`?client_id=`**CLIENT_ID**`&client_secret=`**CLIENT_SECRET**`&ll=`**LATITUDE**`,`**LONGITUDE**`&v=`**VERSION**

# We want to look at the foot traffic around Umi Restaurant and get the trending venues around Umi.

# In[ ]:


# define URL
url = 'https://api.foursquare.com/v2/venues/trending?client_id={}&client_secret={}&ll={},{}&v={}'.format(CLIENT_ID, CLIENT_SECRET, latitude, longitude, VERSION)

# send GET request and get trending venues
results = requests.get(url).json()
results


# In[ ]:


# check if there are any venues trending at this time.
if len(results['response']['venues']) == 0:
    trending_venues_df = 'No trending venues are available at the moment!'
    
else:
    trending_venues = results['response']['venues']
    trending_venues_df = json_normalize(trending_venues)

    # filter columns
    columns_filtered = ['name', 'categories'] + ['location.distance', 'location.city', 'location.postalCode', 'location.state', 'location.country', 'location.lat', 'location.lng']
    trending_venues_df = trending_venues_df.loc[:, columns_filtered]

    # filter the category for each row
    trending_venues_df['categories'] = trending_venues_df.apply(get_category_type, axis=1)


# In[ ]:


# display trending venues
trending_venues_df


# In[ ]:


# lets visualize 
if len(results['response']['venues']) == 0:
    venues_map = 'Cannot generate visual as no trending venues are available at the moment!'

else:
    venues_map = folium.Map(location=[latitude, longitude], zoom_start=15) # generate map centred around Ecco


    # add Ecco as a red circle mark
    folium.features.CircleMarker(
        [latitude, longitude],
        radius=10,
        popup='Umi',
        fill=True,
        color='red',
        fill_color='red',
        fill_opacity=0.6
    ).add_to(venues_map)


    # add the trending venues as blue circle markers
    for lat, lng, label in zip(trending_venues_df['location.lat'], trending_venues_df['location.lng'], trending_venues_df['name']):
        folium.features.CircleMarker(
            [lat, lng],
            radius=5,
            poup=label,
            fill=True,
            color='blue',
            fill_color='blue',
            fill_opacity=0.6
        ).add_to(venues_map)
        
# display map
venues_map


# # Conclusion
# 
# Based on our data analysis we dont have great feedback from the users in regards to a decent Japanese Restaurant. All the Japanese restaurants within the area are not rated. However, we are able to see from the User Feedback that Sariku Restaurant is a good candiate for a decent Japanese food. 
