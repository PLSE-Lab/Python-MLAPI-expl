#!/usr/bin/env python
# coding: utf-8

# Data Source: https://www.kaggle.com/worldbank/world-development-indicators <br> Folder: 'world-development-indicators'

# # Using Folium Library for Geographic Overlays
# 
# ### Further exploring CO2 Emissions per capita in the World Development Indicators Dataset
# 

# In[1]:


import folium
import pandas as pd


# ### Country coordinates for plotting
# 
# source: https://github.com/python-visualization/folium/blob/master/examples/data/world-countries.json

# In[2]:


country_geo = '../input/python-folio-country-boundaries/world-countries.json'


# In[3]:


# Read in the World Development Indicators Database
data = pd.read_csv('../input/world-development-indicators/Indicators.csv')
data.shape


# In[4]:


data.head()


# Pull out CO2 emisions for every country in 2011

# In[5]:


# select CO2 emissions for all countries in 2011
hist_indicator = 'CO2 emissions \(metric'
hist_year = 2011

mask1 = data['IndicatorName'].str.contains(hist_indicator) 
mask2 = data['Year'].isin([hist_year])

# apply our mask
stage = data[mask1 & mask2]
stage.head()


# ### Setup our data for plotting.  
# 
# Create a data frame with just the country codes and the values we want plotted.

# In[6]:


plot_data = stage[['CountryCode','Value']]
plot_data.head()


# In[7]:


# label for the legend
hist_indicator = stage.iloc[0]['IndicatorName']


# ## Visualize CO2 emissions per capita using Folium
# 
# Folium provides interactive maps with the ability to create sophisticated overlays for data visualization

# In[8]:


# Setup a folium map at a high-level zoom @Alok - what is the 100,0, doesn't seem like lat long
map = folium.Map(location=[100, 0], zoom_start=1.5)


# In[10]:


# choropleth maps bind Pandas Data Frames and json geometries.  This allows us to quickly visualize data combinations
map.choropleth(geo_data=country_geo, data=plot_data,
             columns=['CountryCode', 'Value'],
             key_on='feature.id',
             fill_color='YlGnBu', fill_opacity=0.7, line_opacity=0.2,
             legend_name=hist_indicator)


# In[ ]:


# Create Folium plot
map.save('plot_data.html')


# In[ ]:


# Import the Folium interactive html file
from IPython.display import HTML
HTML('<iframe src=plot_data.html width=700 height=450></iframe>')


# More Folium Examples can be found at:<br>
# https://folium.readthedocs.io/en/latest/quickstart.html#getting-started <br>
# 
# Documentation at:<br>
# https://media.readthedocs.org/pdf/folium/latest/folium.pdf

# In[ ]:




