#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import Modules

import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.basemap import Basemap
import warnings
warnings.filterwarnings('ignore')
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode()
import folium


# In[2]:


def load_data(data):
    data_file = pd.read_csv(data, sep = ",", low_memory = False, encoding = "L5")
    return data_file


# In[3]:


data = load_data("../input/meteorite-landings.csv")
data = data[(data.reclat != 0.0) & (data.reclong != 0.0)]
clean_data = data.dropna()


# In[4]:


clean_data.replace(to_replace="Fell", value = "Seen", inplace=True)
df = clean_data.groupby(["fall", "year"]).size().unstack().T.fillna(0).astype(int)
df = df.reset_index()
df = df[df.year >= 1900][df.year <= 2016]


# In[5]:


df.plot(x="year", y="Seen", figsize=(8,4))
plt.title("Meteorites seen per year 1900-2016")
plt.tight_layout()


# In[6]:


df.plot(x="year", y="Found", figsize=(8,4))
plt.title("Meteorites found per year 1900-2016")
plt.tight_layout()


# In[7]:


clean_data2 = clean_data[clean_data.year >= 1900][clean_data.year <= 2016][clean_data.fall == "Found"]


# In[8]:


plt.figure(figsize=(10,10))
ax = clean_data2.plot(kind='scatter', x='reclong', y='reclat',color="red", s=15, alpha=0.3)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.title("Meteorite founded from 1990 to 2016")
plt.tight_layout


# In[9]:


plt.figure(figsize=(8,8))
ax = clean_data2.plot(kind='scatter', x='year', y='reclat',color="red",s=15,alpha=0.3,ax=plt.gca())
ax.set_xlabel("Year")
ax.set_ylabel("Latitude")


# In[10]:


clean_data2['text'] = clean_data2['year'].astype(str) + ' Year, ' +                    clean_data2['mass'].astype(str) + ' grams()'
                    
data = dict(
            type = "scattergeo",
            locationmode = "world",
            lon = clean_data2["reclong"],
            lat = clean_data2["reclat"],
            text = clean_data2["text"],
            mode = "markers",
            name = "Found",
            hoverinfo = "text+name", 
            marker = dict(
                    
                    opacity = 0.5,
                    color = "red")
            )

layout = dict(
         title = 'Meteorite founded from 1900 to 2016',
         showlegend = True,
         legend = dict(
             x = 0.85, y = 0.4
         ),
         geo = dict(
            resolution = 50,
            scope = 'world',
            showframe = True,
            showcoastlines = True,
            showcountries = True,
            showland = True,
            countrywidth=1,
            landcolor = "rgb(229, 229, 229)",
            countrycolor = "rgb(255, 255, 255)" ,
            coastlinecolor = "rgb(255, 255, 255)",
            projection = dict(type = 'Mercator'),
            lonaxis = dict( range= [ -30.0, 50.0 ] ),
            lataxis = dict( range= [ 20.0, 75.0 ] ),
            domain = dict(
                x = [ 0, 1 ],
                y = [ 0, 1 ]
            )))

figure = dict( data = [data], layout = layout )
iplot(figure)
plt.show()


# In[ ]:


map_osm = folium.Map(location = [55.0, 0.0],zoom_start=3.5)
lat = clean_data2.reclat
long = clean_data2.reclong
import math
for x,y in zip(lat,long):
    if math.isnan(x) == False:
        folium.Marker([x,y]).add_to(map_osm)
map_osm

