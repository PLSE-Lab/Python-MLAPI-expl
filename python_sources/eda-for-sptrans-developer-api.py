#!/usr/bin/env python
# coding: utf-8

# # EDA for SPTrans Developer API - Group 21
# 
# **This notebook performs some Exploratory Data Analysis over SPTrans Developer API. All the tables are in a SQL database, bus.db.**

# <em>Importing libraries</em>

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import pickle
import folium
from folium.plugins import HeatMap
import sqlite3
import plotly.express as px
import os

pd.options.display.max_columns = None
pd.options.display.max_rows = 300


# In[ ]:


data_dir = os.path.abspath('../input/so-paulo-bus-system/')


# **Importing some tables**

# In[ ]:


overview = pd.read_csv(os.path.join(data_dir, 'overview.csv'))
overview.head()


# In[ ]:


overview.info()


# In[ ]:


trips = pd.read_csv(os.path.join(data_dir, 'trips.csv'))


# In[ ]:


trips.head()


# In[ ]:


trips.info()


# In[ ]:


routes = pd.read_csv(os.path.join(data_dir, 'routes.csv'))
routes.head()


# In[ ]:


len(routes['route_id'].unique())


# In[ ]:


stops = pd.read_csv(os.path.join(data_dir, 'stops.csv'))
stops.head()


# In[ ]:


stops.info()


# In[ ]:


import numpy as np


# In[ ]:


stops['stop_desc'] = stops['stop_desc'].apply(lambda x: x if x != None else np.nan)


# In[ ]:


len(stops['stop_id'].unique())


# ### Stops Visualization
# 
# **Now that we have the data frames, let's plot some maps to see the bus_stops**

# <em>Function for generating new folium map</em>

# In[ ]:


def generate_base_map(default_location=[-23.5489, -46.6388],default_zoom_start=11,):
    base_map = folium.Map(location=default_location, control_scale=True, zoom_start=default_zoom_start)
    return base_map


# In[ ]:


folium_map = generate_base_map()


# <em>Here we are getting only the unique stops. It will make the map lighter.</em>

# In[ ]:


# Getting unique stops
unique_stops = stops.drop_duplicates(['stop_id'])

# Generating map
for i in range(len(unique_stops)):
    marker = folium.CircleMarker(location=[unique_stops['stop_lat'][i], unique_stops['stop_lon'][i]], radius = 1, color='r', fill=True)
    marker.add_to(folium_map)


# In[ ]:


folium_map


# **Now let's plot a heatmap of the stops!**
# 
# <em>Here we are going to use all the stops in the stops data frame. Doing so, we can also weight the concentration of bus routes in a certain stop.</em>

# In[ ]:


stops['count'] = 1
base_map = generate_base_map()
HeatMap(data=stops[['stop_lat', 'stop_lon', 'count']].groupby(['stop_lat', 'stop_lon']).sum().reset_index().values.tolist(), radius=8, max_zoom=15).add_to(base_map)
base_map


# Pretty interesting! We candefinetly see some stops hotspots. There are also some areas that do not have much bus stops!

# **Let's import one more table to plot the shape of the bus routes**

# In[ ]:


shapes = pd.read_csv(os.path.join(data_dir, 'shapes.csv'))


# In[ ]:


shapes.shape


# In[ ]:


shapes.head()


# <em>Transforming coordinates in a tuple</em>

# In[ ]:


shapes['shape_coords'] = shapes.apply(lambda x: (x['shape_pt_lat'], x['shape_pt_lon']), axis=1)


# In[ ]:


shapes['shape_coords'].head()


# *Let's set a random color to each line so we can visualize better*

# In[ ]:


import random
def random_color():
    a = random.randint(0,256)
    b = random.randint(0,256)
    c = random.randint(0,256)
    rgbl=[a,b,c]
    random.shuffle(rgbl)
    return tuple(rgbl)

def genhex():
    rgb = random_color()
    return '#%02x%02x%02x' % rgb


# *Genrating new map*

# In[ ]:


folium_map = generate_base_map()


# *Creating lines with PolyLines*

# In[ ]:


for shape in list(shapes.groupby('shape_id')):
    df = shape[1]
    marker = folium.PolyLine(locations=df['shape_coords'].to_list(), color=genhex())
    marker.add_to(folium_map)


# *This would be the code for adding the first and last stop of the route, but the map gets too laggy.*

# In[ ]:


'''
for shape in list(shapes.groupby('shape_id')):
    df = shape[1]
    coord_list = df['shape_coords'].to_list()
    initial_point = coord_list[0]
    terminal_point = coord_list[len(coord_list)-1]
    route_edges = [initial_point, terminal_point]
    
    for point in route_edges:
        marker = folium.Marker(location=[point[0], point[1]])
        marker.add_to(folium_map)
'''
0


# In[ ]:


folium_map


# In[ ]:


overview.tail(80)


# ## Stops Stats

# In[ ]:


stops_quantity = overview.groupby('trip_id').count()
stops_quantity['index'].describe()


# In[ ]:


stops_quantity.rename(columns = {'index':'stops_quantity'}, inplace=True)


# In[ ]:


px.histogram(stops_quantity, x='stops_quantity', histnorm='density', labels={'stops_quantity':'Number of Stops'})


# It seems unnatural that a bus line has more than 80 lines. Now imagine having 132 lines! That is impressive. Let's see how these routes are displayed in map

# ### Plotting stops with more than 80 stops

# In[ ]:


stops_quantity.reset_index(inplace=True)


# Merging with the overview dataset

# In[ ]:


stops_quantity.head()


# In[ ]:


stops_quantity_merger = stops_quantity[['trip_id','stops_quantity']].copy()


# In[ ]:


overview_stops_qnt = overview.merge(stops_quantity_merger, on='trip_id', how='outer')


# Dropping the unnecessary rows (Duplicates of stop_id) for plotting the map

# In[ ]:


shapes_stops_qnt = shapes.merge(overview_stops_qnt.drop_duplicates(['shape_id']), on='shape_id', how='outer')


# In[ ]:


shapes_stops_qnt.drop(['index_x', 'index_y'], axis=1, inplace=True)


# In[ ]:


many_stops_shapes = shapes_stops_qnt[shapes_stops_qnt['stops_quantity'] >= 80]


# *Generating map*

# In[ ]:


folium_map = generate_base_map()


# *Generating lines*

# In[ ]:


for shape in list(many_stops_shapes.groupby('shape_id')):
    df = shape[1]
    marker = folium.PolyLine(locations=df['shape_coords'].to_list(), color=genhex())
    marker.add_to(folium_map)


# *Plotting Map*

# In[ ]:


folium_map


# ### Plotting lines with less than 5 stops

# In[ ]:


few_stops_shapes = shapes_stops_qnt[shapes_stops_qnt['stops_quantity'] <= 10]


# *Generating Map*

# In[ ]:


folium_map = generate_base_map()


# *Generating lines*

# In[ ]:


for shape in list(few_stops_shapes.groupby('shape_id')):
    df = shape[1]
    marker = folium.PolyLine(locations=df['shape_coords'].to_list(), color=genhex())
    marker.add_to(folium_map)


# *Plotting Map*

# In[ ]:


folium_map


# In[ ]:


overview.head()


# ## Correlations

# In[ ]:


overview.corr()


# **Let's get the frequencies table**

# In[ ]:


stop_times = pd.read_csv(os.path.join(data_dir, 'stop_times.csv'))


# In[ ]:


stop_times.head()


# In[ ]:


frequencies = pd.read_csv(os.path.join(data_dir, 'frequencies.csv'))


# In[ ]:


frequencies.head()


# *Example of frequencies for one bus line*

# In[ ]:


frequencies[frequencies['trip_id'] == '1012-10-0']


# In[ ]:


overview.head()


# ## Passengers Analysis

# In[ ]:


passengers = pd.read_csv(os.path.join(data_dir, 'passengers.csv'))
passengers.head()


# ### Too heavy too run

# In[ ]:


# bus_position = pd.read_csv(os.path.join(data_dir, 'bus_position.csv'))
# bus_position.head()


# In[ ]:


# bus_position.tail()

