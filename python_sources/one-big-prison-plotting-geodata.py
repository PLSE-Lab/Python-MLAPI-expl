#!/usr/bin/env python
# coding: utf-8

# What associations appear in your mind when you hear word Russia? Maybe bears, or borscht, or rough men with balalaikas? I don't know what about you, but in my case Russia associated with prison - this is the country where the freedom is not exists. So, in this short kernel I want to fulfill my curiosity and plot prisons on Russia map.
# 
# Less talk, more actions.

# In[8]:


# I'll ude bokeh for data visualisation
import pandas as pd
import pyproj
from bokeh.plotting import figure, show
from bokeh.tile_providers import get_provider, Vendors
from bokeh.io import output_notebook
from bokeh.models import ColumnDataSource

# Make bokeh show plot in jupyter notebook
output_notebook() 


# In[9]:


# Reading our data
data = pd.read_csv('../input/Rus_prisons_coords.csv', encoding = 'windows-1251')
data.drop('Unnamed: 0', inplace = True, axis = 1)


# In[10]:


# Preparing coordinates
lat = data['lat'].values
lon = data['lon'].values


# Before we start to plot our data, we need to do some preparations. I scraped coordinates from google maps, but bokeh uses different projection - UTM, so we need to transform our coordinates from google projection to UTM. 

# In[12]:


# Transforming coordinates
project_projection = pyproj.Proj("+init=EPSG:4326")  # wgs84
google_projection = pyproj.Proj("+init=EPSG:3857")  # default google projection
x, y = pyproj.transform(project_projection, google_projection, lon, lat)

# Plotting data
p = figure(x_range=(2000000, 18000000), y_range=(6000000, 11000000),
            plot_width=800, plot_height=500, x_axis_type="mercator", y_axis_type="mercator")
p.add_tile(get_provider(Vendors.CARTODBPOSITRON_RETINA))
p.circle(x=x, y=y, size=3, fill_color="blue")
show(p)


# I can see very familiar things here - the picture is very similar to what I saw, when plotted 1k (100M of ~146M of people) of biggest cities in Russia (maybe I'll publish it sometime). The main part of prisons located in south-west part of Russia. It's a little bit surprising for me, because I expected to see a lot of such facilities in northern part of Russia.
# We also can see a couple of prisons in occupied by Russia Crimea.
