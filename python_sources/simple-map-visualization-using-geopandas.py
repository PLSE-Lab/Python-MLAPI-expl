#!/usr/bin/env python
# coding: utf-8

# <img src="https://cdn.pixabay.com/photo/2015/04/09/16/39/indonesia-714747_960_720.png" width=150 height=130 align=center />

# # Simple map visualization using geopandas <br>
# 
# Hi there, data lovers! You'll learn how to create map-based visualizations using the geopandas library.
# geopandas makes pandas even cooler by adding spatial operation functionalities!
# The dataset used here is the __[geospatial data of Indonesian provinces](https://github.com/superpikar/indonesia-geojson)__ joined with the __[2010 population by BPS](https://www.bps.go.id/statictable/2009/02/20/1267/penduduk-indonesia-menurut-provinsi-1971-1980-1990-1995-2000-dan-2010.html)__.
# 
# Enjoy, <br>
# fariz@ui.ac.id

# In[ ]:


import geopandas as gpd

filename = '../input/indonesiaprovincejmlpenduduk/indonesia-province-jml-penduduk.json' # population data for irian jaya timur and irian jaya tengah is taken from that of papua province divided by two
df = gpd.read_file(filename)
print(type(df))
df # take a look at the whole data


# In[ ]:


df.plot() # plot the whole Indonesia, x-axis is for longitude, y-axis is for latitude


# In[ ]:


df.set_index("Propinsi", inplace=True) # so you can get a specific row by its province
df.head() # return first 5 rows, look how Propinsi is now set as index (leftmost column)


# In[ ]:


df.loc['JAWA TIMUR'] # show only the specific data of JAWA TIMUR


# In[ ]:


series = gpd.GeoSeries(df.loc['JAWA TIMUR']['geometry']) # get the geometry of JAWA TIMUR and convert it into GeoSeries
type(series)


# In[ ]:


ax = series.plot() # make a plot of JAWA TIMUR
ax


# **Quiz: Make a map plot of DKI Jakarta**

# In[ ]:


series = gpd.GeoSeries(df.loc['DKI JAKARTA']['geometry']) # get the geometry of DKI Jakarta and convert it into GeoSeries
type(series)
ax = series.plot() # make a plot of DKI Jakarta
ax


# **Choropleth map of Indonesia based on population** <br>
# FYI, a __[choropleth map](https://en.wikipedia.org/wiki/Choropleth_map)__ is a thematic map type where areas are shaded or patterned in proportion to the measurement of the statistical variable being displayed on the map, such as population, population density, or per-capita income.

# In[ ]:


import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)
df = df[df['Jumlah Penduduk'].notnull()] # pick rows where Jumlah Penduduk is not null, just in case
df.plot(column='Jumlah Penduduk', ax=ax, legend=True)


# In[ ]:


# in case you are not that fond of the above color scheme, let's redo it with another color scheme
# also this time with a bigger size
fig, ax = plt.subplots(1, 1, figsize=(20, 10))
df.plot(column='Jumlah Penduduk', ax=ax, legend=True, cmap='OrRd')


# **Congrats! You did it!** <br>
# 
# Once you are able to visualize choropleth maps, you may simpy adapt the technique to other problems, such as transport traffic visualization, forest fire visualization, economy disparity visualization, and so on. <br>
# 
# ~Bye for now and have fun!
