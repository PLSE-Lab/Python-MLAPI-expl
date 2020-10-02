#!/usr/bin/env python
# coding: utf-8

# **[Geospatial Analysis Home Page](https://www.kaggle.com/learn/geospatial-analysis)**
# 
# ---
# 

# # Introduction
# 
# In this tutorial, you'll learn about two common manipulations for geospatial data: **geocoding** and **table joins**.

# In[ ]:



import pandas as pd
import geopandas as gpd
import numpy as np
import folium
from folium import Marker


# # Geocoding
# 
# **Geocoding** is the process of converting the name of a place or an address to a location on a map.  If you have ever looked up a geographic location based on a landmark description with [Google Maps](https://www.google.com/maps), [Bing Maps](https://www.bing.com/maps), or [Baidu Maps](https://map.baidu.com/), for instance, then you have used a geocoder!
# 
# ![](https://i.imgur.com/1IrgZQq.png)
# 
# We'll use geopandas to do all of our geocoding.

# In[ ]:


from geopandas.tools import geocode


# To use the geocoder, we need only provide: 
# - the name or address as a Python string, and
# - the name of the provider; to avoid having to provide an API key, we'll use the [OpenStreetMap Nominatim geocoder](https://nominatim.openstreetmap.org/).
# 
# If the geocoding is successful, it returns a GeoDataFrame with two columns:
# - the "geometry" column contains the (latitude, longitude) location, and
# - the "address" column contains the full address.

# In[ ]:


result = geocode("The Great Pyramid of Giza", provider="nominatim")
result


# The entry in the "geometry" column is a `Point` object, and we can get the latitude and longitude from the `y` and `x` attributes, respectively.

# In[ ]:


point = result.geometry.iloc[0]
print("Latitude:", point.y)
print("Longitude:", point.x)


# It's often the case that we'll need to geocode many different addresses.  For instance, say we want to obtain the locations of 100 top universities in Europe.

# In[ ]:


universities = pd.read_csv("../input/geospatial-learn-course-data/top_universities.csv")
universities.head()


# Then we can use a lambda function to apply the geocoder to every row in the DataFrame.  (We use a try/except statement to account for the case that the geocoding is unsuccessful.)

# In[ ]:


def my_geocoder(row):
    try:
        point = geocode(row, provider='nominatim').geometry.iloc[0]
        return pd.Series({'Latitude': point.y, 'Longitude': point.x, 'geometry': point})
    except:
        return None

universities[['Latitude', 'Longitude', 'geometry']] = universities.apply(lambda x: my_geocoder(x['Name']), axis=1)

print("{}% of addresses were geocoded!".format(
    (1 - sum(np.isnan(universities["Latitude"])) / len(universities)) * 100))

# Drop universities that were not successfully geocoded
universities = universities.loc[~np.isnan(universities["Latitude"])]
universities = gpd.GeoDataFrame(universities, geometry=universities.geometry)
universities.crs = {'init': 'epsg:4326'}
universities.head()


# Next, we visualize all of the locations that were returned by the geocoder.  Notice that a few of the locations are certainly inaccurate, as they're not in Europe!

# In[ ]:


# Create a map
m = folium.Map(location=[54, 15], tiles='openstreetmap', zoom_start=2)

# Add points to the map
for idx, row in universities.iterrows():
    Marker([row['Latitude'], row['Longitude']], popup=row['Name']).add_to(m)

# Display the map
m


# # Table joins
# 
# Now, we'll switch topics and think about how to combine data from different sources.  
# 
# ### Attribute join
# 
# [You already know](https://www.kaggle.com/residentmario/renaming-and-combining) how to use `pd.DataFrame.join()` to combine information from multiple DataFrames with a shared index.  We refer to this way of joining data (by simpling matching values in the index) as an **attribute join**.
# 
# When performing an attribute join with a GeoDataFrame, it's best to use the `gpd.GeoDataFrame.merge()`.  To illustrate this, we'll work with a GeoDataFrame `europe_boundaries` containing the boundaries for every country in Europe.  The first five rows of this GeoDataFrame are printed below.

# In[ ]:



world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
europe = world.loc[world.continent == 'Europe'].reset_index(drop=True)

europe_stats = europe[["name", "pop_est", "gdp_md_est"]]
europe_boundaries = europe[["name", "geometry"]]


# In[ ]:


europe_boundaries.head()


# We'll join it with a DataFrame `europe_stats` containing the estimated population and gross domestic product (GDP) for each country.

# In[ ]:


europe_stats.head()


# We do the attribute join in the code cell below.  The `on` argument is set to the column name that is used to match rows in `europe_boundaries` to rows in `europe_stats`.

# In[ ]:


# Use an attribute join to merge data about countries in Europe
europe = europe_boundaries.merge(europe_stats, on="name")
europe.head()


# ### Spatial join
# 
# Another type of join is a **spatial join**.  With a spatial join, we combine GeoDataFrames based on the spatial relationship between the objects in the "geometry" columns.  For instance, we already have a GeoDataFrame `universities` containing geocoded addresses of European universities.  
# 
# Then we can use a spatial join to match each university to its corresponding country.  We do this with `gpd.sjoin()`.

# In[ ]:


# Use spatial join to match universities to countries in Europe
european_universities = gpd.sjoin(universities, europe)

# Investigate the result
print("We located {} universities.".format(len(universities)))
print("Only {} of the universities were located in Europe (in {} different countries).".format(
    len(european_universities), len(european_universities.name.unique())))

european_universities.head()


# The spatial join above looks at the "geometry" columns in both GeoDataFrames.  If a Point object from the `universities` GeoDataFrame intersects a Polygon object from the `europe` DataFrame, the corresponding rows are combined and added as a single row of the `european_universities` DataFrame.  Otherwise, countries without a matching university (and universities without a matching country) are omitted from the results.
# 
# The `gpd.sjoin()` method is customizable for different types of joins, through the `how` and `op` arguments.  For instance, you can do the equivalent of a SQL left (or right) join by setting `how='left'` (or `how='right'`).  We won't go into the details in this micro-course, but you can learn more in [the documentation](http://geopandas.org/reference/geopandas.sjoin.html).
# 
# # Your turn
# 
# **[Use geocoding and table joins](https://www.kaggle.com/kernels/fork/5832170)** to identify suitable locations for the next Starbucks Reserve Roastery.

# ---
# **[Geospatial Analysis Home Page](https://www.kaggle.com/learn/geospatial-analysis)**
# 
# 
# 
# 
# 
# *Have questions or comments? Visit the [Learn Discussion forum](https://www.kaggle.com/learn-forum/161464) to chat with other Learners.*
