# %% [markdown]
# # Hurricane Michael
# 
# Download the hurricaneMichael.csv and US_states(5m).json files.
# 
# Using the geopandas library, turn the latitude and longitude columns into a geographical Point data type then make a geodataframe. Plot the path of Hurricane Michael onto the US map in the GeoJSON file.
# 
# Tips
# 
# *    After loading the US_states(5m).json file as a geodataframe, use the following code to create a geodataframe that only contains the contiguous United States (48 states):
# 

# %% [raw]
#  map48 = map_df.loc[map_df['NAME'].isin(['Alaska', 'Hawaii', 'Puerto Rico']) == False]
#     

# %% [markdown]
# *    The longitude column data should be turned into negative values(data source listed longitude direction instead of positive/negative). Use the following code to make the data correct:
# 
# Feel free to add any additional features to your plot (marker shape, marker color, etc.).

# %% [raw]
# michaeldf['Long'] = 0 - michaeldf['Long']

# %% [code]
import pandas as pd
import geopandas as gpd #used for transforming geolocation data
import matplotlib.pyplot as plt

from datetime import datetime  #to convert data to datetime that does not fall within the pandas.to_datetime function timeframe
from shapely.geometry import Point  #transform latitude/longitude to geo-coordinate data
from geopandas.tools import geocode #get the latitude/longitude for a given address
from geopandas.tools import reverse_geocode  #get the address for a location using latitude/longitude

#%matplotlib inline

# %% [markdown]
# ### Geocoding and Reverse Geocoding
# 
# Geocoding is taking an address for a location and returning its latitudinal and longitudinal coordinates. Reverse geocoding would then be the opposite - taking the latitudinal and longitudinal coordinates for a location and returning the physical address.

# %% [code] {"scrolled":true}
#load hurricane data collected 
hurricane_df = pd.read_csv("../input/hurricaneMichael.csv")
hurricane_df.head()

# %% [code]
hurricane_df['Long'] = 0 - hurricane_df['Long']

# %% [code]
#data type of each column
hurricane_df.dtypes

# %% [code]
len(hurricane_df)

# %% [code]
#see columns with null values
hurricane_df.count()

# %% [code]
#make a new column to hold the longitude & latitude as a list
hurricane_df['coordinates'] = list(hurricane_df[['Long', 'Lat']].values)

# %% [code] {"scrolled":true}
#see new coordinates column
hurricane_df.head()

# %% [code]
#list values in coordinates column is classified as object type
hurricane_df['coordinates'].dtypes

# %% [code]
#convert the coordinates to a geolocation type
hurricane_df['coordinates'] = hurricane_df['coordinates'].apply(Point)

# %% [code] {"scrolled":true}
#coordinates column now has POINT next to each coordinate pair value
hurricane_df.head()

# %% [code]
#coordinates column with geolocation data is just a regular pandas Series type
type(hurricane_df['coordinates'])

# %% [code]
#create a geolocation dataframe type using the coordinates column as the geolocation data
geo_hurricane = gpd.GeoDataFrame(hurricane_df, geometry='coordinates')

# %% [code]
#geo-dataframe looks the same as regular dataframe
geo_hurricane.head()

# %% [code]
#verify coordinates column is geolocation data type
type(geo_hurricane['coordinates'])

# %% [code]
#import file that contains a US map shape polygons
#will use to plot the coordinates of meteorite landings
filepath = "../input/US_states(5m).json"

#data contains polygon shape coordinates for different map body types (states, etc.)
map_df = gpd.read_file(filepath)
map48 = map_df.loc[map_df['NAME'].isin(['Alaska', 'Hawaii', 'Puerto Rico']) == False]
map48.head()

# %% [code]
#map graph
map48.plot(cmap='OrRd')

# %% [code]
#plot the coordinates (no map)
geo_hurricane.plot()

# %% [code]
#plot coordinates on top of map graph

#this is to set the size of the borders
fig, ax = plt.subplots(1, figsize=(10,15))

#this is the map , cmap='OrRd'
basemap = map48.plot(ax=ax,  cmap='OrRd')

#plot coordinates on top of map graph
geo_hurricane.plot(ax=basemap, color='black', marker="o", markersize=50)

#take off axis numbers
ax.axis('off')

#put title on map
ax.set_ylim([17, 58])
ax.set_title("Hurrican Michael Trajectory", fontsize=25, fontweight=3)

# %% [markdown]
# ### We could zoom-in and see the trajectory 

# %% [code]
#plot coordinates on top of map graph

#this is to set the size of the borders
fig, ax = plt.subplots(1, figsize=(10,15))

#this is the map , cmap='OrRd'
basemap = map48.plot(ax=ax,  cmap='OrRd')

#plot coordinates on top of map graph
geo_hurricane.plot(ax=basemap, color='black', marker=">", markersize=50)

#take off axis numbers
ax.axis('off')

#put title on map
ax.set_xlim([-90, -75])
ax.set_ylim([17.5, 37.5])
ax.set_title("Hurrican Michael Trajectory", fontsize=25, fontweight=3)

# %% [code]
