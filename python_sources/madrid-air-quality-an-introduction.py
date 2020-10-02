#!/usr/bin/env python
# coding: utf-8

# As my first dataset to analysis I chose the Madrid Air Quality dataset uploaded here on the site.
# At first we should import the necessary libraries.
# 
# **For the compact version of the code, please visit my GitHub https://github.com/chrischris96/Madrid-Air-Quality **

# In[ ]:


import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from shapely.geometry import Point, LineString


# In the dataset which you can find easily on kaggle.com, I take the location of the stations and save the .csv in a variable. 

# In[ ]:


path = r'../input/air-quality-madrid/stations.csv'


# In[ ]:


stations = pd.read_csv(path)


# In the following I take extract a tuple of longitude and latitude points and save them into a new variable. The following lines are unintuitive if you have never used geopandas .However there are different ways to map a card using different projection methods. So we use one specific projection for the European continent called epsg:4326. For every other card we use in the following, the same projection has to be appplied. 

# In[ ]:


geometry = [Point(xy) for xy in zip(stations['lon'], stations['lat'])]
crs = {'init': 'epsg:4326'}
geoDF_stations = gpd.GeoDataFrame(stations, crs=crs, geometry=geometry)
geoDF_stations_new = geoDF_stations.to_crs({'init': 'epsg:25830'}) 


# The variable geoDF_stations_new contains all the points with its coordinates. Next, we want to load a streetmap of Madrid and load it into a variable. Since streets are not specified by only one tuple of coordinates, a so called linestring represents a street. A linestring consists of multiple coordinates with in sum can be connected to a line which leads us to the form of the street. However, the streetmap is from http://www.madrid.org/nomecalles/DescargaBDTCorte.icm which is down at the monent, but I found https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=a4f36d34fa86c410VgnVCM2000000c205a0aRCRD& to be a good source too.
# 
# **Sadly, .shp cannot be read by the interpreter but if you download the data, no error will occur and  a plot like the one you see in the last image will be plotted**

# In[ ]:


#streetsystem = gpd.read_file('../input/location-of-the-streets-and-stations/call2016.shp')


# In[ ]:


#calleselected = streetsystem.loc[streetsystem['VIA_TVIA'] == "Calle"]
#avdselected = streetsystem.loc[streetsystem['VIA_TVIA'] == "Avda"]
#ctraselected = streetsystem.loc[streetsystem['VIA_TVIA'] == "Ctra"]
#calleandavd = calleselected.append(avdselected)
#streetselected = calleandavd.append(ctraselected)


# As you see, I only took the most relevant streets into account since the map is overloaded otherwise. The most frequent street names in Madrid begin with "Calle", "Avenia" or "Carretera" . In the following we will take the station parameter we got from the .csv file and lay it over the street map of Madrid. 

# In[ ]:


#base = geoDF_stations_new.plot(figsize=(32,20), marker='o',color='red',markersize=100.0,label='Stations');
#mapMadrid = streetselected.plot(figsize=(32,20), ax=base,color='blue', edgecolor='blue',markersize=0.01,label='Streets');


# Now, we adjust the plot and add a title, legend, label the axes and choose a specific frame since Madrid is huge and the stations are just in a specific area, i.e. the city center. Finally the map can be plotted.

# In[ ]:


#plt.ylim((4465000,4485000))
#plt.xlim((430000,455000))
#plt.legend(loc = 'lower right', framealpha=1)
#plt.xlabel("Longitude")
#plt.ylabel("Latitude")
#plt.title("Madrid city center street map with measurement stations")


# Finally the map can be plotted.

# In[ ]:


#plt.show(mapMadrid)


# ![madrid3.png](attachment:madrid3.png)
