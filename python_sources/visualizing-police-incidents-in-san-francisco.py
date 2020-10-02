#!/usr/bin/env python
# coding: utf-8

# # Working with Geospatial Data: Spatial Joins and Creating Maps

# ### Introduction
# 
# In this notebook we are going to deal with data describing police incidents in the city of San Francisco from 2003-2018. It includes details such as category of the incident (e.g. Assault, Theft, etc.), a more detailed description of the incident, resolution of the incident(e.g. the person was arrested), date/time of the incident, and a rough address where the incident took place.
# 
# On its own, the provided data set would allow us for example to analyze police incidents by perhaps aggregating the incidents by category and study its distribution over various time ranges.
# 
# However, and crucially, the data includes exact coordinates (longitude, latitude) of the police incident. This opens up **a lot** of avenues for EDA/Modelling/Visualization!
# 
# If we can find geospatial data that describes the boundaries of geographical areas, we can analyze these police incidents at a very finely granular level with respect to its geography. Moreover we can potentially connect this data to other rich data sources by linking to the applicable geographic subdivision.
# 
# Lucky for us, the government of San Francisco has a very large public data repository. We will use this by downloading so-called *shapefiles* containing geographical boundaries for each census tract in San Francisco. A census tract is one of the smallest subdivisions for which data is collected in the United States. Over at the US Census website there are truly enormous amounts of data broken down by census tract that, once we know the census tract for each police incident, could greatly inform our analysis. For now, though, all we need is:
# 1. The Police Incidents in SF data (it's here on kaggle!)
# 2. A shapefile containing all the census tracts of San Francisco (from data.sf.gov)
# 3. A shapefile containing all the neighborhoods of San Francisco (optional; I use this to showcase the map-making capabilities; also from data.sf.gov)
# 
# ### Goals
# 
# * Creating geospatial dataframes with the help of **geopandas**
# * Performing spatial joins of two dataframes
# * Creating some cool maps with **geoplot**!
# 
# Let's start by importing the required libraries:

# In[ ]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# handles geodata
import geopandas as gp
# converts coordinate pairs into points that can be interpreted by geopandas
from shapely.geometry import Point
# map plotting
import geoplot as gplt
import geoplot.crs as gcrs
# geoplot is based on cartopy
import cartopy
import cartopy.crs as ccrs


# ### Importing Shapefiles as Geodataframes
# 
# Next, we load the census tract boundaries into a *geodataframe* which, quite intuitively, is the *spatial* equivalent of a regular pandas dataframe.

# In[ ]:


# reading the shapefile
sftracts = gp.read_file("../input/shapefiles-for-sf/sf_tracts_and_neighborhoods.dbf")

# we don't need these columns in this notebook
sftracts.drop(columns=["geoid", "shape_area", "shape_len"], inplace=True)

# setting the census tract ID as index will make map creation easier later on!
sftracts.set_index("tractce10", inplace=True)

sftracts.head()


# Looks almost like a plain dataframe with one obvious exception. Every geodataframe contains a *geometry*. In particular, every row in the geodataframe is assigned either a point (when we only have a single longitude/latitude pair) or a polygon (like here, where we have a succession of longitude/latitude pairs that are connected by a straight line, starting with the first coordinate pair).
# 
# Note: The neighborhoods are made up of several census tracts. The census tracts are unique.
# 
# Let's repeat the same thing, this time loading neighborhood boundaries:

# In[ ]:


sfnhoods = gp.read_file("../input/shapefiles-for-sf/sf_nhoods.dbf")
sfnhoods.set_index("nhood", inplace=True)

sfnhoods.head(5)


# ### Converting the Police Incident Data into a Geodataframe
# 
# Great! Now we need to load the police incident data set. The *location* column is rather useless, as far as I can tell, so I exclude it from the csv reader by passing a function to the *usecols* argument.
# 
# This dataset is not a geospatial dataset (yet!), so we need to do some minor prep-work to be able to convert it into a geodataframe. The *X* and *Y* columns hold the coordinates (lon/lat) and need to be combined into tuples before we can proceed which *apply(tuple, axis=1)* takes care of. Then we need to convert these tuples to points so that geopandas can make sense of them.

# In[ ]:


file = "../input/sf-police-calls-for-service-and-incidents/police-department-incidents.csv"

#read file without Location column
sf_incidents = pd.read_csv(file, usecols=lambda x: x not in ["Location"],
                           dtype={"IncidntNum":str})

# convert to datetime and merge date and time columns
sf_incidents["Date"] = pd.to_datetime(sf_incidents.Date) 
sf_incidents["Date"] = pd.to_datetime(sf_incidents.Date.dt.date.astype(str)
                                      + " " + sf_incidents.Time)
sf_incidents.drop("Time", axis=1, inplace=True)

# convert coords to points
sf_incidents["Coordinates"] = (sf_incidents[["X", "Y"]]
                               .apply(tuple, axis=1)
                               .apply(Point)
                              ) 
# convert dataframe to geodataframe
sf_incidents = gp.GeoDataFrame(sf_incidents, geometry="Coordinates")

sf_incidents.head()


# ### Spatial Joining of Dataframes
# 
# We now have converted a regular dataframe into a geodataframe! Note that the geometry of this dataframe consists of points this time around, rather than polygons.
# 
# Let us focus on these two dataframes for now:
# * *sftracts* contains area polygons of census tracts and the name of its associated neighborhood
# * *sf_incidents* contains points of police incidents along with all the data that describes these incidents
# 
# When we join regular dataframes, we join them on some shared key-value.
# Here, our keys are the geometries, that is we join the points on the polygon. What that means is for every point in the police incident data, the spatial joining procedure checks whether the point lies in any of the given census tract areas (or polygons to be precise). If it does, the data is matched and joined!
# 
# **Note: This next block of code contains the actual spatial join function. Unfortunately, at this point it will not work in a kaggle notebook, as a dependency is missing (libspatialindex) that to the best of my knowledge can not be installed currently. As a workaround, I will simply upload and import the resulting dataframe of the spatial join!
# When working on your local setup everything should work. In that case please uncomment this next block, and skip the workaround!**

# In[ ]:


# # Uncomment this when working locally!
# sf_incidents.crs = sftracts.crs # Making sure the map projections of both geodataframes are the same
# sf_incidents = gp.sjoin(sf_incidents, sftracts) # joining the geodataframes on its spatial geometries
# sf_incidents.rename(columns={"index_right":"tractce10"}, inplace=True)
# sf_incidents.iloc[:,::-1].head()


# ### Workaround (Skip this in a local setup):

# In[ ]:


file2 = "../input/spatiallyjoineddata/police_incidents_after_spatial_join.csv"
sf_incidents = pd.read_csv(file2, parse_dates=[4], dtype={"tractce10":str})

# Restore the leading zeros in this column (optional)
sf_incidents["IncidntNum"] = sf_incidents.IncidntNum.apply(lambda x: "%09d" % x)

# Annoyingly we have to recreate the geometry from scratch
sf_incidents["Coordinates"] = sf_incidents[["X", "Y"]].apply(tuple, axis=1).apply(Point) 
sf_incidents = gp.GeoDataFrame(sf_incidents, geometry="Coordinates")

sf_incidents.iloc[:,::-1].head()


# We now have successfully matched each police incident with a census tract and a neighborhood in San Francisco. Just over a 1000 police incidents have been dropped during the spatial join. I looked into this a little bit and found that either:
# 
# - these were incidents on the edge of the city limits but just outside; 
# - or wrong/default coordinates that are nowhere near the city; 
# - or the incident took place on a bridge between two census tracts, the bridge not being part of any census tract.
# 
# However, 1000 observations out of over two million is a very tiny fraction of the information being lost, so I don't think we have to be too bothered by it.
# 
# Moving on, we will retrieve a subsample from the data, specifically all incidents during the year 2015, with which we will work going forward:

# In[ ]:


sf_incidents_sub = sf_incidents.loc[sf_incidents.Date.dt.year==2015]


# Which neighborhood had the most police incidents in 2015?

# In[ ]:


(sf_incidents_sub.nhood.value_counts()
 .to_frame().rename(columns={"nhood":"Incidents"}).head(5))


# And which census tract?

# In[ ]:


(sf_incidents_sub.tractce10.value_counts()
 .to_frame().rename(columns={"tractce10":"Incidents"}).head(5))


# Great! We are now able to aggregate our data by geographic subdivisions, where before the best we could do was aggregate by police district, which are quite large and therefore don't allow for a more detailed inspection of crime patterns in San Francisco.
# 
# Of course, the real value of having census tract information lies in being able to connect the police data to the rich U.S. census data including the American Community Survey (ACS). We could analyze crime patterns by very detailed demographics such as age, income, race, household, population density, relationship status, and many more!
# 
# This is, however, not going to be part of this notebook. Instead, in the second part I want to show you how we can create some really cool maps with just a few lines of code using geoplot.
# 
# ### Mapping Police Incidents by Geographic Subdivision with Geoplot
# 
# First, we need (actually, we don't *need* - more on that in a bit) to aggregate our data. For a simple example we will count the number of drug-related incidents in each census tract:

# In[ ]:


drug_inc_by_tract = gp.GeoDataFrame((sf_incidents_sub
                                     .loc[sf_incidents_sub["Category"] == "DRUG/NARCOTIC"]
                                     .tractce10.value_counts()
                                     .to_frame()
                                     .rename(columns={"tractce10":"Incidents"})
                                    ).merge(sftracts.geometry.to_frame(),
                                            left_index=True, right_index=True)
                                   )
drug_inc_by_tract.head()


# Aggregating that data is a bit awkward and cumbersome since during aggregation the geo-properties are lost and have to be restored by merging with the census tract shapefile and turning it back into a geodataframe. Fortunately, I will later show that there is a way to avoid this intermediate step, as geoplot will handle it automatically. For illustrative purposes we will create our first map "by hand". You will also find that when using some more involved aggregation schemes (as opposed to just counting, for instance), the method of manually aggregating the data and then plotting the map based on that aggregated data, can be significantly more performant.
# 
# That annoyance aside, we can now visualize this data by creating a map, which as you'll surely agree in a moment is surprisingly simple.
# Here, we will create a map that colors each census tract by the number of incidents. We can use any color palette contained in matplotlib.
# 
# We start with an axis object created by the *geoplot.polyplot()* function. The polyplot function simply plots all the shapes that we provide as an argument to the function. In this case, we want to draw the census tract boundaries contained in *sftracts.geometry*. 
# 
# Before we do this, I'd like to talk just a little bit about map projections!
# 
# ### Quick Aside: Map Projections
# 
# We will also provide the *projection* argument to specify the map projection. The map projection determines how the surface of our globe (which obviously is a three-dimensional object) is projected onto a two-dimensional plane (the map). It is mathematically impossible to project a map without some form of distortion, and there are always trade-offs between the types of distortion (e.g. area, shapes, distance and direction). To illustrate the differences, here are some possible options we can choose. They are all plotted with *cartopy*. Geoplot is a high-level implementation of cartopy.

# In[ ]:


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
plt.style.use("seaborn-white")

fig = plt.figure(figsize=(12,10))
fig.suptitle("Various Map Projections", fontweight="bold", fontsize=20)

ax1 = plt.subplot(221, projection=ccrs.Mollweide())
ax1.coastlines()
ax1.stock_img()
ax1.set_anchor("N")
ax1.set_title("Mollweide Projection", fontweight="bold")

ax2=plt.subplot(222,projection=ccrs.Mercator())
ax2.coastlines()
ax2.stock_img()
ax2.set_anchor("N")
ax2.set_title("Mercator Projection", fontweight="bold")

ax3 = plt.subplot(223, projection=ccrs.InterruptedGoodeHomolosine())
ax3.coastlines()
ax3.stock_img()
ax3.set_anchor("N")
ax3.set_title("Goode Homolosine Projection", fontweight="bold")

ax4 = plt.subplot(224, projection=ccrs.AlbersEqualArea())
ax4.coastlines()
ax4.stock_img()
ax4.set_anchor("N")
ax4.set_title("Albers Equal Area Projection", fontweight="bold")
plt.show()


# These are just a small subset of the possible options, each with its set of advantages and disadvantages depending on the intended use of the map. Since I am no expert in cartography I had to rely on the expertise of the internet in choosing the appropriate projection type.
# 
# Apparently for middle latitudes (neither close to the poles nor to the equator) and for local maps the *Lambert Conformal Projection* is a solid choice! The Mercator projection also would have worked well for local-scale maps (for world maps it is a pretty bad, though often used, choice).
# 
# ### Creating a Map of the Census Tract Boundaries - geoplot.polyplot()
# 
# Back to our actual objective: mapping crime data!
# 
# This first map will eventually consist of two layers. Our first layer are the census tract boundaries, which we will draw with the geoplot.polyplot() function, as mentioned and explained in detail above. We have our geometric information and we settled on the appropriate map projection. That's all we need for now so let's draw our first map!

# In[ ]:


ax = gplt.polyplot(sftracts.geometry, projection = gcrs.LambertConformal(),
                   figsize=(12,8), edgecolor="k")
ax.set_title("Census Tract Boundaries in San Francisco", fontweight="bold", fontsize=16)
plt.show()


# That sure looks like San Francisco (note all the piers in the northeast of the city)! For this step, all we really needed were the shapes (in this case census tract boundaries) stored in the shapefile. Drawing this simple map would have already involved quite a bit of busywork in cartopy (advantage: you have full control over everything). For example, geoplot automatically sets the extents of the map and centers it based on the shapes provided. Most annoyingly we would have had to re-project the coordinates in our geodataframe, since the way cartopy works with coordinate systems is slightly different from geopandas.
# 
# Now we draw another layer on top of this map, which will color the census tracts by the number of police incidents. The function for this job will be *geoplot.choropleth()*.
# 
# ### Coloring Census Tracts by Drug-Related Police Incidents - geoplot.choropleth()
# 
# This function needs a geodataframe (the aggregated police data), a map projection, the variable to color by (the number of police incidents), and how many bins of data there should be (passing *None* creates a continuous colormap). We then layer this map on top of the map we created just before.
# 
# **Note**: Throughout this notebook, when I am aggregating police incidents, I will count every person involved in the incident as a separate case. To illustrate, if two persons rob a store and get arrested, I will count this as two incidents. If you disagree with this approach, your aggregation function should incorporate unique Incident IDs.

# In[ ]:


ax = gplt.choropleth(drug_inc_by_tract, projection=gcrs.LambertConformal(), hue="Incidents",
                     cmap="magma_r", k=None, linewidth=0, figsize=(10,8), legend=True)
gplt.polyplot(sftracts.geometry, projection = gcrs.LambertConformal(), edgecolor="k",
              linewidth=1, ax=ax)
ax.set_title("2015 SF Police Incidents by Census Tract\nCategory: Drug/Narcotic",
             fontweight="bold", fontsize=16)
plt.show()


# Maps can be super effective visualizations of data. Just a cursory glance reveals to us where narcotics crimes are rampant. If you are familiar with San Francisco, you will immediately recognize the darkest area as the Tenderloin neighborhood just north of Market Street. It is one of the lowest income areas of San Francisco and a major hotbed of crime. It is also within walking distance of the major downtown shopping and commercial areas, so I'm sure many a tourist has accidentally walked into this neighborhood.
# 
# ### Simplifying the Map Creation Process - geoplot.aggplot()
# 
# That was pretty simple, all things considered, wasn't it? But wait, we can simplify this process even more! The *geoplot.aggplot()* function will take care of the data aggregation for us, which is super handy and saves us some busywork (or defining our own set of functions).
# 
# The best part: the geometric shapes and the data we want to visualize don't even need to be in the same dataframe. All we need is a key variable in the data we want to visualize, so it can match the index in the geodataframe where the shapes are stored. We took care of that earlier by pulling the census tract numbers into the police dataframe.
# 
# We need to pass this function:
# - a dataframe (our 2015 subsample of the police incident data, which we can filter)
# - a map projection
# - the variable to be aggregated (in our case any will do, since we are just counting)
# - the aggregation function (here: simply len())
# - the geographic entity we will aggregate by (this needs to match the index of the geodataframe, where the geometry is located)
# - the location where the geometry is stored (here: sftracts.geometry)
# 
# We are going to mix it up a little by using a different category of crime and a different colormap!

# In[ ]:


ax = gplt.aggplot(sf_incidents_sub.loc[sf_incidents_sub["Category"] == "ASSAULT"],
                  projection=gcrs.LambertConformal(),
                  hue="Date", agg=len, by="tractce10",
                  geometry=sftracts, cmap="coolwarm", linewidth=0, figsize=(12,8),
                  vmin=0, vmax = 400)
gplt.polyplot(sftracts.geometry, projection=gcrs.LambertConformal(),
              linewidth=1, edgecolor="k", ax=ax)
ax.set_title("2015 SF Police Incidents by Census Tract\nCategory: Assault",
             fontweight="bold", fontsize=16)
plt.show()


# Let's add another layer on top! After all, we also have a geodataframe containing neighborhood boundaries. We will aggregate total incidents by neighborhood this time but preserve the census tract boundaries as well.
# 
# This time we'll pass a custom aggregation function and log-transform (base 10 log) the data. This gives us a better visualization of the order of magnitude of police incident in each neighborhood.

# In[ ]:


ax = gplt.aggplot(sf_incidents_sub,
                  projection=gcrs.LambertConformal(),
                  hue="PdId", agg=lambda x: np.log10(len(x)), by="nhood",
                  geometry=sfnhoods.geometry, cmap="viridis", linewidth=0, figsize=(12,8),
                  vmin=2, vmax=4)
gplt.polyplot(sftracts.geometry, gcrs.LambertConformal(), ax=ax, linewidth=0.5, edgecolor="w")
gplt.polyplot(sfnhoods.geometry, gcrs.LambertConformal(), ax=ax, edgecolor="k")
ax.set_title("2015 SF Police Incidents (Total) by Neighborhood\n Log-Transformed",
             fontweight="bold", fontsize=16)
plt.show()


# This is actually a pretty neat visualization, as it shows us clearly three low-incident outliers in the northwest of the city. So naturally, at this point you would ask yourself (having no prior knowledge of the San Francisco Police Department like myself) what the reason for this is. Doing some really quick research yields that going from left to right:
# - one is essentially just a park with most of its space covered by a golf course
# - two is a *very* exclusive little oceanview neighborhood for the rich and famous
# - and three is a large park (around the Golden Gate bridge area), that very importantly is administered by the National Park Service and not by the City of San Francisco. The National Park Service has their own police and therefore is not part of the SF Police Department's jurisdiction. So even if a ton of crime were to take place there, it would not show in this dataset
# 
# Obviously, when analyzing this dataset, you would probably want to take that into account one way or the other!
# 
# ### Visualizing Changes in Incidents Year-Over-Year
# 
# Next, and finally, we will use a customized aggregation function and use that data to map the percentage change in police incident between two years within a given category (or total incidents). The first function will take a grouped dataframe and calculate the percentage change in incidents from a base year to another year. The second function creates a dataframe by using the aggregation method from the first function. This data will then be used to create a map:

# In[ ]:


def crime_change(g, t0=2014, t1=2015): # calculates percentage changes
    g = g.dt.year
    a = len(g.loc[g==t1])
    b = len(g.loc[g==t0])
    if b > 0:
        c = 100*(a-b)/b
    elif ((b == 0) & (a > 0)):
        c = np.nan
    else: 
        c = 0
    return c

def crime_change_data(data, t0=2014, t1=2015, category=None, geography="nhood"):
    shapes = {"nhood": sfnhoods, "tractce10": sftracts}
    if category is None:
        g = data.groupby(geography).Date
    else:
        g = data.loc[data["Category"]==category].groupby(geography).Date
    return (gp.GeoDataFrame(g.apply(crime_change, t0=t0, t1=t1)
                           .to_frame().rename(columns={"Date":str(t0)+str(t1)+"change"})
                           )
            .join(shapes[geography], how="right")
           )


# Let's use this to create a dataframe we will pass to the choropleth function in a minute. We will look at the % change between the years 2016 and 2017 in total incidents (category=None) by census tract. We will drop all observations that happened in the neighborhood of Presidio:

# In[ ]:


crime_change1617 = (crime_change_data(sf_incidents.loc[sf_incidents["nhood"] != "Presidio"],
                                     t0=2016, t1=2017, geography="tractce10", category=None)
                    .dropna())


# Plotting the three layers (census tracts, coloring, neighborhoods):

# In[ ]:


ax = gplt.polyplot(sftracts.geometry, gcrs.LambertConformal(), facecolor="lightgrey",
                   figsize=(12,8), linewidth=0)
gplt.choropleth(crime_change1617, gcrs.LambertConformal(), hue="20162017change",
                ax=ax, k=None, cmap="RdBu_r", edgecolor="k", linewidth=0.5,
                vmin=-30, vmax=30, legend=True, legend_kwargs={"extend":"both"})
gplt.polyplot(sfnhoods.geometry, gcrs.LambertConformal(), ax=ax, edgecolor="k")
ax.set_title("% Change in SF Total Incidents by Census Tract\n2016 - 2017",
             fontweight="bold", fontsize=16)
plt.show()


# A pretty mixed picture here: Some of the more problematic areas downtown have seen a slight improvement, but neighboring areas are experiencing some pretty sharp increases. Perhaps the problem is merely shifting from one place to another?
# 
# And that shall be it for this notebook. I hope you had as much fun as I had discovering and playing around with tools such as geopandas and geoplot!
# 
# ### Where could you go from here?
#  - Merging the crime data with readily available census data to get a more detailed picture of each census tract and neighborhood
#  - Determining incident rates (dividing the number of incidents by the population in each geographic subdivision) to obtain a more comparable measure
#  
# ### Summary
# In this notebook we have:
# 
# - Learned how to handle shapefiles containing geographic boundaries
# - Learned how to convert a dataframe into a geodataframe
# - Learned how to merge datasets based on geographic intersections
# - Learned how to visualize data geographically by creating maps
# 
# **Thank you for reading!**

# In[ ]:




