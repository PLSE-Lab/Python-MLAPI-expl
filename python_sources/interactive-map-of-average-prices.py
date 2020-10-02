#!/usr/bin/env python
# coding: utf-8

# The purpose of this notebook is to provide a visual overview (interactive map) of the average prices throughout the areas (neighborhoods) of Iowa.

# Importing libraries and markdown

# In[19]:


import numpy as np
import pandas as pd
import folium
import os
from IPython.display import display, HTML
train = pd.read_csv('../input/train.csv')


# With the help of http://geojson.io, we created a json file, containing the coordinates of the Iowa map as a polygon. Let us import that file

# In[20]:


iowa_poly = os.path.join('../input/iowa_poly.json')


# Next, we create a folium map, centered in Iowa.

# In[21]:


m = folium.Map(location = [42.0564052,-93.6442311], zoom_start = 7)


# The next step we do is manually create a dataframe, containing the shortname of each neighborhood, along with the full name and coordinates of the neighborhood. The coordinates were found in Google Maps.
# 
# More specifically, each area was typed in Google Maps and the coorninates were obtained from the URL. Then they were manually added to the DF creation.

# In[22]:


coords_v0 = pd.DataFrame(np.array(
        [
        ['Blmngtn','Bloomington Heights',42.0564052,-93.6442311]
       ,['Blueste','Bluestem',41.497925,-93.5011687]
       ,['BrDale','Briardale',42.5228147,-93.2860814]
       ,['BrkSide','Brookside',42.028952,-93.6319627]
       ,['ClearCr','Clear Creek',41.8199517,-93.3600346]
       ,['CollgCr','College Creek',42.0214541,-93.6671637]
       ,['Crawfor','Crawford',41.377671,-93.9140169]
       ,['Edwards','Edwards',42.0154064,-93.6875441]
       ,['Gilbert','Gilbert',42.1068336,-93.6553512]
       ,['IDOTRR','Iowa DOT and Rail Road',42.0220014,-93.6242068]
       ,['MeadowV','Meadow Village',42.0048434,-93.6568125]
       ,['Mitchel','Mitchell',43.3185572,-92.8779557]
       ,['NAmes','North Ames',42.059172,-93.6441717]
       ,['NoRidge','Northridge',42.0485012,-93.6526078]
       ,['NPkVill','Northpark Villa',42.0499088,-93.6290747]
       ,['NridgHt','Northridge Heights',42.0597767,-93.6500184]
       ,['NWAmes','Northwest Ames',42.042906,-93.6642637]
       ,['OldTown','Old Town',43.3135899,-95.1529172]
       ,['SWISU','South & West of Iowa State University',42.0318986,-93.6585304]
       ,['Sawyer','Sawyer',40.6964434,-91.3639064]
       ,['SawyerW','Sawyer West',40.706617,-91.3805747]
       ,['Somerst','Somerset',41.5233969,-93.6141585]
       ,['StoneBr','Stone Brook',42.059385,-93.6355362]
       ,['Timber','Timberland',41.720651,-91.4766478]
       ,['Veenker','Veenker',42.0416438,-93.6513107]
]), columns = ['Neighborhood','Neighborhood_FULL','Lat','Long']) 

coords_v0["Lat"] = pd.to_numeric(coords_v0["Lat"])
coords_v0["Long"] = pd.to_numeric(coords_v0["Long"])


# Let us now calculate the average price of houses per neighborhood, as it is given to us by the training dataset. The average price will be merged with neighborhood DF that we just created.

# In[23]:


mean_price_per_neighborhood = train.groupby('Neighborhood', as_index=False)['SalePrice'].mean()
mean_price_per_neighborhood['SalePrice'] = round(mean_price_per_neighborhood['SalePrice'])

coords = pd.merge(coords_v0, mean_price_per_neighborhood, on = 'Neighborhood')
display(HTML(coords.to_html()))


# We will get the quantiles of the mean_price_per_neighborhood DF. We will obtain four prices. More specifically:
# * The minimum price of the DF
# * The 50th percentile
# * The 75th percentile 
# * And the maximum average house price
# 
# This way, we are able to classify which areas have low, mediumn and high house prices.

# In[24]:


p_min = mean_price_per_neighborhood['SalePrice'].describe()['min']
p_50 = mean_price_per_neighborhood['SalePrice'].describe()['50%']
p_75 = mean_price_per_neighborhood['SalePrice'].describe()['75%']
p_max = mean_price_per_neighborhood['SalePrice'].describe()['max']


# The get_color() function will make the map more dynamic. The color of the marker for each neighborhood will depend on its percentile position in the mean_price_per_neighborhood DF.

# In[25]:


def get_color(x):        
    if (p_min <= x < p_50):
        return "../input/green.png"
    elif(p_50 <= x <= p_75):
        return "../input/blue.png"
    elif(p_75 < x <= p_max):
        return "../input/red.png" 


# With a for loop we add the markers to the map. We are fully using the <b>mean_price_per_neighborhood dataframe</b>. We are specifying the <b>latitude</b> and <b>longitude</b> of the marker. Also, the <b>coloring</b> and the <b>pop-ups</b> depend on the average price of houses in the neighborhood.

# In[26]:


for index, row in coords.iterrows():
    icon_path = get_color(row['SalePrice'])
    logo_icon = folium.features.CustomIcon(icon_image=icon_path ,icon_size=(18,18))
    folium.Marker([row['Lat'],row['Long']]
    ,popup = row['Neighborhood_FULL']+'\n AVG_Price:' +str(row['SalePrice'])
    ,icon=logo_icon).add_to(m)


# In[27]:


folium.GeoJson(iowa_poly, name = 'iowa_poly').add_to(m)


# In[28]:


m


# Every marker on the map represents a neighborhood of our dataset. Clicking on a marker displays the name of the neighborhood and the average price of houses in that neighborhood. The color represents whether the average price is low, medium or high.
