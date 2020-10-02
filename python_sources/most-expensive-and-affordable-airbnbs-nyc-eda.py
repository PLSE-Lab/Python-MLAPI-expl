#!/usr/bin/env python
# coding: utf-8

# This kernel explores the New York city Airbnb data to find out the following:
# 1. Most expensive Airbnb in each neighbourhood. 
# 2. Affordable and good quality airbnbs within 1 mile of World Trade Center, Manhattan. 
# 
# Utilizing Google Maps API (Gmaps, API needed) to visualize these airbnbs alongwith more details when you click on the airbnb. Gmaps doesn't show up the visualization on this kernel, I have included the pics separately to see how the output looks.
# 
# 

# In[ ]:


import datetime
import os
import numpy as np
import pandas as pd
from scipy import stats
from pandas.io import gbq
import geopandas as gpd
from shapely import wkt
import matplotlib.pyplot as plt
import urllib
import gmaps
import json
from IPython.display import Image
import os
get_ipython().system('ls ../input')


# In[ ]:


data = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
data.info()


# ## Most Expensive Airbnbs in each Neighbourhood

# In[ ]:


df_exp = data.groupby(['neighbourhood_group'])[["price", "host_id", "name", "neighbourhood", "room_type", "latitude",
                                                "longitude", "minimum_nights", "number_of_reviews"]]\
.agg(lambda x: {
    "price": x["price"].max(),
    "host_id": x.loc[x["price"]==x["price"].max(), "host_id"].iloc[0],
    "name": x.loc[x["price"]==x["price"].max(), "name"].iloc[0],
    "neighbourhood": x.loc[x["price"]==x["price"].max(), "neighbourhood"].iloc[0],
    "room_type": x.loc[x["price"]==x["price"].max(), "room_type"].iloc[0],
    "latitude": x.loc[x["price"]==x["price"].max(), "latitude"].iloc[0],
    "longitude": x.loc[x["price"]==x["price"].max(), "longitude"].iloc[0],
    "minimum_nights": x.loc[x["price"]==x["price"].max(), "minimum_nights"].iloc[0],
    "number_of_reviews": x.loc[x["price"]==x["price"].max(), "number_of_reviews"].iloc[0]
})


# In[ ]:


df_exp.head()


# In[ ]:


plt.figure(figsize=(18,15))
url=urllib.request.urlopen('https://upload.wikimedia.org/wikipedia/commons/e/ec/Neighbourhoods_New_York_City_Map.PNG')
map_nyc=plt.imread(url)
plt.imshow(map_nyc,zorder=0,extent=[-74.258, -73.7, 40.49,40.92])
ax=plt.gca()
df_exp.plot(kind='scatter', x='longitude', y='latitude', c='price', ax=ax, 
           cmap=plt.get_cmap('Spectral'), colorbar=True, alpha=1.0, zorder=10, s=80)
for i, txt in enumerate(df_exp.room_type):
    ax.annotate(txt, (df_exp.longitude[i], df_exp.latitude[i]), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
for i, txt in enumerate(df_exp.neighbourhood):
    ax.annotate(txt, (df_exp.longitude[i], df_exp.latitude[i]), ha='right', va='bottom', color='blue', 
               alpha=1.0, fontsize=15, weight='bold')
plt.title("Most Expensive Airbnb For Each Neighbourhood")
plt.show()


# The above shows the type and neighbourhood of most expensive airbnbs in the given 5 neighbourhoods of the New York city. The color of the dots show the price range.

# ## Affordable Airbnb Options Near 4World Trade Center

# In[ ]:


from math import radians, cos, sin, asin, sqrt

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r


# In[ ]:


data.boxplot(column='price', by='neighbourhood_group', figsize=(12,8), showfliers=False, grid=False)


# Affordable if below the median price of each neighbourhood, after removing the outliers

# In[ ]:


df_median = data.groupby(['neighbourhood_group'])['price'].median()


# In[ ]:


data_aff = data.loc[data.apply(lambda x: x["price"] <= df_median.loc[x["neighbourhood_group"]], axis=1)]


# In[ ]:


data_aff.groupby(['neighbourhood_group', 'room_type'])['id'].count()


# In[ ]:


office_coord = [{'lat': 40.7103, 'lng': -74.0123}]

def return_in_out(x):

    aff_airbnb_coord = [{'lat': x.latitude, 'lng': x.longitude}]

    lat1 = office_coord[0]['lat']
    lon1 = office_coord[0]['lng']
    lat2 = aff_airbnb_coord[0]['lat']
    lon2 = aff_airbnb_coord[0]['lng']

    radius = 1.6 # in kilometer

    a = haversine(lon1, lat1, lon2, lat2)

    if a <= radius:
        return 'Inside'
    return 'Outside'


# In[ ]:


data_aff['range'] = data_aff.apply(return_in_out, axis=1)


# In[ ]:


df_within_range = data_aff[data_aff.range=='Inside']
df_within_range.shape


# In[ ]:


df_within_range_good_quality = df_within_range[df_within_range.number_of_reviews>df_within_range.number_of_reviews.mean()]
df_within_range_good_quality.shape


# In[ ]:


df_within_range_good_quality.groupby(['room_type'])['id'].count()


# ### Good Quality, within a mile affordable shared rooms

# In[ ]:


df_shared = df_within_range_good_quality[df_within_range_good_quality.room_type=='Shared room']


# In[ ]:


all_info = (df_shared[["name", "price", "number_of_reviews", "neighbourhood", 'latitude', 'longitude']].to_dict(orient="records"))
all_info = [((x["latitude"], x["longitude"]), x) for x in (all_info)]


# In[ ]:


airbnb_lat_long = [x[0] for x in all_info]


info_box_template = """
<dl>
<dt>Name</dt><dd>{name}</dd>
<dt>Price</dt><dd>{price}</dd>
<dt>Number of reviews</dt><dd>{number_of_reviews}</dd>
<dt>Neighbourhood</dt><dd>{neighbourhood}</dd>
</dl>
"""

airbnb_info = [info_box_template.format(**info) for info in ([x[1] for x in all_info])]


# In[ ]:


office_location = gmaps.symbol_layer([(40.7103, -74.0123)], fill_color='blue', stroke_color='blue', scale=6)
shared_airbnbs = gmaps.symbol_layer(airbnb_lat_long, 
                                 fill_color='red', stroke_color='red', scale=3, info_box_content=airbnb_info)
fig = gmaps.figure()
fig.add_layer(office_location)
fig.add_layer(shared_airbnbs)
fig


# In[ ]:


Image("../input/gmaps-airbnb/Screen Shot 2019-09-21 at 12.00.46 PM.png")


# ### Good Quality, within a mile affordable Private room

# In[ ]:


df_shared = df_within_range_good_quality[df_within_range_good_quality.room_type=='Private room']


# In[ ]:


all_info = (df_shared[["name", "price", "number_of_reviews", "neighbourhood", 'latitude', 'longitude']].to_dict(orient="records"))
all_info = [((x["latitude"], x["longitude"]), x) for x in (all_info)]


# In[ ]:


airbnb_lat_long = [x[0] for x in all_info]


info_box_template = """
<dl>
<dt>Name</dt><dd>{name}</dd>
<dt>Price</dt><dd>{price}</dd>
<dt>Number of reviews</dt><dd>{number_of_reviews}</dd>
<dt>Neighbourhood</dt><dd>{neighbourhood}</dd>
</dl>
"""

airbnb_info = [info_box_template.format(**info) for info in ([x[1] for x in all_info])]


# In[ ]:


office_location = gmaps.symbol_layer([(40.7103, -74.0123)], fill_color='blue', stroke_color='blue', scale=6)
shared_airbnbs = gmaps.symbol_layer(airbnb_lat_long, 
                                 fill_color='red', stroke_color='red', scale=3, info_box_content=airbnb_info)
fig = gmaps.figure()
fig.add_layer(office_location)
fig.add_layer(shared_airbnbs)
fig


# In[ ]:


Image("../input/gmaps-airbnb/Screen Shot 2019-09-21 at 12.01.43 PM.png")


# ### Good Quality, within a mile affordable Entire Apt

# In[ ]:


df_shared = df_within_range_good_quality[df_within_range_good_quality.room_type=='Entire home/apt']


# In[ ]:


all_info = (df_shared[["name", "price", "number_of_reviews", "neighbourhood", 'latitude', 'longitude']].to_dict(orient="records"))
all_info = [((x["latitude"], x["longitude"]), x) for x in (all_info)]


# In[ ]:


airbnb_lat_long = [x[0] for x in all_info]


info_box_template = """
<dl>
<dt>Name</dt><dd>{name}</dd>
<dt>Price</dt><dd>{price}</dd>
<dt>Number of reviews</dt><dd>{number_of_reviews}</dd>
<dt>Neighbourhood</dt><dd>{neighbourhood}</dd>
</dl>
"""

airbnb_info = [info_box_template.format(**info) for info in ([x[1] for x in all_info])]


# In[ ]:


office_location = gmaps.symbol_layer([(40.7103, -74.0123)], fill_color='blue', stroke_color='blue', scale=6)
shared_airbnbs = gmaps.symbol_layer(airbnb_lat_long, 
                                 fill_color='red', stroke_color='red', scale=3, info_box_content=airbnb_info)
fig = gmaps.figure()
fig.add_layer(office_location)
fig.add_layer(shared_airbnbs)
fig


# In[ ]:


Image("../input/gmaps-airbnb/Screen Shot 2019-09-21 at 12.01.55 PM.png")


# In[ ]:




