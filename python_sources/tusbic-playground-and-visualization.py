#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas


# In[ ]:


import numpy


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot
from mpl_toolkits import basemap


# In[ ]:


import shapely
import shapely.wkt


# # Loading and preparing the data
# 
# Lets load the data and see how it looks

# In[ ]:


data = pandas.read_csv('../input/bikes.csv')


# In[ ]:


data.head(4)


# In[ ]:


data.info()


# In[ ]:


data.describe()


# ## Data cleanup
# 
# The above output shows some data that might be wrong, as the maximum for the column `available_bike_stands` is larger than `bike_stands`. This last column is the number of stands that there are at a given station, so we cannot have more bikes or more available stands than `bike_stands`, see:

# In[ ]:


data[["bike_stands", "available_bike_stands", "available_bikes"]].describe()


# Let's drop those rows from our dataset.

# In[ ]:


data = data[data["bike_stands"] >= (data["available_bike_stands"] + data["available_bikes"])]


# In[ ]:


data.describe()


# Now lets check if there is something else we can drop. It seems that the `status`, `banking` and `bonus` values are good candidates, as they give us little or no information at all.

# In[ ]:


data.status.value_counts()


# In[ ]:


data.banking.value_counts()


# In[ ]:


data.bonus.value_counts()


# In[ ]:


data = data.drop(["bonus", "number", "contract_name", "banking", "address"], axis=1)
data.head(3)


# ## Add some other features that might be interesting
# 
# First add a column containing the bike stands that are unusable, due to a broken bike of due to a broken stand

# In[ ]:


data["bad_stands"] = data["bike_stands"] - (data["available_bike_stands"] + data["available_bikes"])


# In[ ]:


data.describe()


# ### Adding time slots
# 
# We are going to classify the station data according to time slots, instead of using the exact timestamp, so we:
# 
# * Convert the timestamp to a datetime
# * Add a new column "time" containing a decimal value (i.e. 16:30 will be 16.5)

# In[ ]:


data["last_update"] = pandas.to_datetime(data["last_update"], unit='ms')
data["time"] = data['last_update'].map(lambda x: x.hour + x.minute / 60)
#data["time"] = data['last_update'].map(lambda x: x.hour)
data.head(3)


# Now we can create our time slots. Bike sharing systems tend to be free for a short period of time (like 1 hour) so we are going to try to group the station information in time clusters.
# 
# Let's try with 20 minutes.

# In[ ]:


data["time_cluster"] = pandas.cut(data["time"], 24 * 3)
data["time_cluster"] = pandas.Categorical(data["time_cluster"]).codes / 3.0


# # Visualize the data

# ## Visualize available bikes for each time of the day

# In[ ]:


grouped = data[["time_cluster", "name", "bike_stands", "available_bikes", "bad_stands"]].groupby(["time_cluster", "name"]).mean()

grouped.head(20)


# In[ ]:


# Reshape the new df
pivot = pandas.pivot_table(grouped.reset_index(), index="time_cluster", columns="name")


# In[ ]:


titles = list(pivot.available_bikes.columns.values)

# Round the times to that we do not show all 20m slots.
indexes = numpy.round(pivot.index)

ax = pivot.available_bikes.plot(subplots=True,
                 grid=True,
                 rot=0,
                 xticks=numpy.round(pivot.index),
                 figsize=(15,60),
                 title=titles,
)
ax = pivot.bad_stands.plot(subplots=True,
                      ax=ax,
                      grid=True,
                      style='r',   
)
ax = pivot.bike_stands.plot(subplots=True,
                      ax=ax,
                      grid=True,
                      style='k--',   
)

for a in ax:
    a.legend(["Available bikes", "Bad stands", "Max stands"])


# ## Visualize stations on a map
# Let's see the stations on a map, drawing the bike lane geometries that we have.

# In[ ]:


bike_lanes_df = pandas.read_csv("../input/bike_lanes.csv")
bike_lanes_df["wkt_wsg84"] = bike_lanes_df["wkt_wsg84"].apply(shapely.wkt.loads)

bike_lanes_df.head(3)


# In[ ]:


bike_lanes = shapely.geometry.MultiLineString(list(bike_lanes_df["wkt_wsg84"]))


# In[ ]:


# define map colors
land_color = '#f5f5f3'
water_color = '#a4bee8'
coastline_color = '#000000'
border_color = '#bbbbbb'

map_width = 10 * 1000
map_height = 7 * 1000

# plot the map
fig_width = 20
fig = pyplot.figure(figsize=(20, 20 * map_height / map_width))

ax = fig.add_subplot(111, facecolor='#ffffff')
ax.set_title("Santander Bike Stations", fontsize=16, color='#333333')

lat = 43.47
lon = -3.82

m = basemap.Basemap(
            projection="tmerc",
            lon_0=lon, 
            lat_0=lat,
            width=map_width, 
            height=map_height,
            resolution='h',
            area_thresh=0.1
)

m.drawmapboundary(fill_color=water_color)
m.drawcoastlines(color=coastline_color)
m.drawcountries(color=border_color)
m.fillcontinents(color=land_color, lake_color=water_color)
m.drawstates(color=border_color)

means = data.groupby("name").mean()

m.scatter(means.lng.values.ravel(), 
          means.lat.values.ravel(),
          latlon=True,
          alpha=0.8,
          s=means["bike_stands"] * 5,
          label="bike stands",
          c=means["available_bikes"].astype(float),
          lw=.25,
          cmap=pyplot.get_cmap("jet"),
          zorder=3
)                

# This should work, but I do not know why it does not
#ax.add_collection(bike_lanes)
for l in bike_lanes:
    m.plot(*l.xy, latlon=True, color="grey", alpha=0.5)

c = pyplot.colorbar(orientation='vertical', shrink=0.5)
c.set_label("Available Bikes")


# Hmm, it seems that the bike lanes are a bit wrong? They are going over the sea! Not really... That portion should be the bike lane that goes through the beach.

# ## Visualize stations on a map (2nd try)
# The previous `basemap` try gave us little resolution, so lets try with another cartopy.

# In[ ]:


import cartopy.crs 
from cartopy.io import img_tiles


# In[ ]:


osm_tiles = img_tiles.OSM()


# In[ ]:


pyplot.figure(figsize=(20, 20))

# Use the tile's projection for the underlying map.
ax = pyplot.axes(projection=osm_tiles.crs)

ax.set_extent([-3.892, -3.762, 43.438, 43.495],
              cartopy.crs.PlateCarree())

# Add the tiles at zoom level 13.
ax.add_image(osm_tiles, 14)

cb = ax.scatter(means.lng.values.ravel(), 
           means.lat.values.ravel(),
           alpha=0.9,
           s=means["bike_stands"] * 5,
           label="bike stands",
           c=means["available_bikes"].astype(float),
           lw=.25,
           cmap=pyplot.get_cmap("jet"),
           zorder=3,
           transform=cartopy.crs.PlateCarree(),
)                

c = pyplot.colorbar(cb, orientation='horizontal', pad=0.04)
c.set_label("Available Bikes")

# This should work, but I do not know why it does not
#ax.add_collection(bike_lanes)
for l in bike_lanes:
    ax.plot(*l.xy, lw=2, color="red", alpha=1, transform=cartopy.crs.PlateCarree())

pyplot.show()


# In[ ]:




