#!/usr/bin/env python
# coding: utf-8

# ### Load the required packages

# In[ ]:


import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import folium


# ### We use two datasets - public art and neighborhoods

# Load the public art data and look at the first few rows it was downloaded from
# https://data.nashville.gov/Art/Art-in-Public-Places/dqkw-tj5j

# In[ ]:


art = pd.read_csv('../input/nashville-public-art/public_art.csv')
art.head()


# Load the neighborhoods data and look at the first few rows. This data was downloaded from 
# https://data.nashville.gov/Metro-Government/Neighborhood-Association-Boundaries-GIS-/qytv-2cu8

# In[ ]:


neighborhoods = gpd.read_file('../input/nashville-nighborhoods/nashville_neighborhoods.geojson')
neighborhoods.head()


# ### Compare *accessing* a geometry to *printing* it

# In[ ]:


neighborhoods.loc[0, 'geometry']


# In[ ]:


print(neighborhoods.loc[0, 'geometry'])


# ###  Plot the neighborhoods. Create a dictionary of legend arguments to give to geopandas to style the legend.

# In[ ]:


# geopandas handles legend styling if you pass a dict of keywords
leg_kwds = {'title': 'Neighborhoods',
               'loc': 'upper left', 'bbox_to_anchor': (1, 1.03), 'ncol': 6}

neighborhoods.plot(column = 'name', legend = True, cmap = 'Set2', legend_kwds = leg_kwds)
plt.title('Neighborhoods')
plt.show();


# ### Make the column names in the art data lowercase with no spaces
# 

# In[ ]:


art.columns = ['title', 'last_name', 'first_name', 'address', 'medium', 'type', 'desc', 'lat', 'lng', 'location']
art.head()


# ### Preparing for a spatial join:  
#  * create a GeoDataFrame from the art DataFrame
#      * create a geometry field
#      * define the coordinate reference system (CRS)

# ### Create a geometry column from lng and lat with help from shapely 

# In[ ]:


art['geometry'] = art.apply(lambda row: Point(row.lng ,row.lat), axis=1)
art.head()


# In[ ]:


type(art)


# ####  Make a GeoDataFrame called `art_geo` from the `art` DataFrame

# In[ ]:


art_geo = gpd.GeoDataFrame(art, crs = neighborhoods.crs, geometry = art['geometry'])


# In[ ]:


type(art_geo)


# ### Join `art_geo` and `neighborhoods` using `.sjoin()` so that they are related spatially to a new GeoDataFrame called `neighborhood_art`. Get the art that is `within` each neighborhood.

# In[ ]:


neighborhood_art = gpd.sjoin(art_geo, neighborhoods, op = 'within')
neighborhood_art.head()


# ### Now that the data is joined spatially we can aggregate the art by neighborhood and see how many artworks are within each polygon

# In[ ]:


neighborhood_art[['name', 'title']].groupby('name').agg('count').sort_values(by = 'title', ascending = False)


# ### Create a GeoDataFrame called `urban_art` by getting only the `neighborhood_art` for the *Urban Residents* neighborhood

# In[ ]:


urban_art = neighborhood_art.loc[neighborhood_art.name == 'Urban Residents']
urban_art.head()


# In[ ]:


urban_art.shape


# ### Next we'll get just the *Urban Residents* polygon from the `neighborhoods` data and take a look at it

# In[ ]:


urban_polygon = neighborhoods.loc[neighborhoods.name == 'Urban Residents']
urban_polygon.head() 


# ### Let's go ahead and plot the *Urban Residents* Neighborhood and add our public art. We'll color the points it by the type of art.

# In[ ]:


# define the plot as ax
ax = urban_polygon.plot(figsize = (12, 12), color = 'lightgreen')

# add the plot of urban_art to ax (the urban_polygon)
urban_art.plot(ax = ax, column = 'type', legend = True);
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.show();


# ### We're getting there! It would be helpful to have streets, though wouldn't it? We can add a street map with the folium package.

# ### Our folium map wants a center point for the street map. We'll make use of an attribute (`.centroid`) that geopandas gets from Shapely to get the center of the urban polygon.

# In[ ]:


# look at the center of our urban_polygon
urban_polygon.geometry.centroid


# ### The centroid  is a pandas series; We can grab the first element which is a Point geometry and store it as `center_point`

# In[ ]:


# find the center of the urban polygon with the centroid property
center = urban_polygon.geometry.centroid

# get and store the first occurence which will be a Point geometry
center_point = center.iloc[0]

# print the types for center and center_point
print('center is :', type(center))
print('center_point is :', type(center_point))


# ### Here's a tricky part! Folium wants a point that is an array that has _latitude_ first, so we build that here by moving the latitude value to the first position in the array

# In[ ]:


# center point has longitude first
print(center_point)

# reverse the order when constructing the array for folium location
urban_center = [center_point.y, center_point.x]

# check the order of urban_center, the location we'll set for our folium map
print(urban_center)


# ### We'll use this urban center as the location of our folium map. A zoom level of 15 should get us nice and close.

# In[ ]:


# create our map of Nashville and show it
map_downtown = folium.Map(location =  urban_center, zoom_start = 15)
map_downtown


# ### We are ready to add the Urban Residents neighborhood polygon and art markers. 

# ###  `itterows()`is a generator that iterates through the rows of a DataFrame and returns a tuple with the row id and row values. Below, we are printing the row values for each row as we iterate through the GeoDataFrame of urban art. This idea will be helpful for creating our markers!

# In[ ]:


# show what iterrows() does
for row in urban_art.iterrows():
    row_values = row[1]
    print(row_values)


# ### The map of Nashville `map_downtown` has already been created. 
# * add the downtown neighborhood outline (`urban_polygon`)
# * iterate through the urban art and 
#     * create `location` from each `lat` and `lng` 
#     * create a `popup` from `title` and `type`
#     * build a `marker` from each `location` and `popup`
#     * add the `marker` to  `map_downtown`
# * display `map_downtown`

# In[ ]:


#draw our neighborhood: Urban Residents
folium.GeoJson(urban_polygon).add_to(map_downtown)

#iterate through our urban art to create locations and markers for each piece
#here lat is listed first!!
#also the apostrophe in the 4th row causes problems!

for row in urban_art.iterrows():
    row_values = row[1] 
    location = [row_values['lat'], row_values['lng']]
    popup = (str(row_values['title']) + ': ' + 
             str(row_values['type']) + '<br/>' +
             str(row_values['desc'])).replace("'", "`")
    marker = folium.Marker(location = location, popup = popup)
    
    marker.add_to(map_downtown)

#display our map
map_downtown


# In[ ]:




