#!/usr/bin/env python
# coding: utf-8

# **<font color='#d64541'>UPDATE</font>**: See my [Final Submission here](https://www.kaggle.com/dsholes/cpe-submission-methodology-and-analysis)
# ## 1. Where do we begin??
# A lot of other kernels jump right into it, but before we start working our Python magic, it's important to understand what this challenge is actually looking for. If you prefer to get right to the code, [click here](#start_of_code). If you're dying to see a map, [click here](#punchline).  BUT, I suggest you start at the beginning. The existing problem statement has a couple of main themes:
# 
# >"Our biggest challenge is automating the **combination of police data, census-level data, and other socioeconomic factors**. Shapefiles are unusual and messy -- which makes it difficult to, for instance, generate maps of police behavior with precinct boundary layers mixed with census layers. Police incident data are also very difficult to **normalize and standardize across departments** since there are no federal standards for data collection.."
# 
# >"How well does the solution **combine shapefiles and census data**?"
# 
# >"How well does it **match census-level demographics to police deployment areas**?"
# 
# So we know we need to focus on **<font color='#d64541'>combining different sources of geospatial data</font>**. Python has a few libraries to help with that. If you look through existing kernels, most are using some combination of `pandas`, `geopandas`,  and `shapely` to handle the **data**, and  `matplotlib`, `plotly` or `folium` to handle the **plotting and visualization** . Now that we have a toolset, we should look through the `cpe-data` to understand what we have to work with.
# 
# ## 2. Data-perusing
# The dataset is quite small (<6MB) so it might be worth it to download the `cpe-data.zip` folder to take a quick look through. Within the `cpe-data` folder, we're given six "Department" folders. Within each "Department" folder we're given at least two sub-directories, one containing Census data (ACS stands for **A**merican **C**ommunity **S**urvey) and one containing [Shapefiles](https://en.wikipedia.org/wiki/Shapefile) that seem to correspond to Police Districts. In three of of the "Department" folders, we're also given a "prepped" CSV that contains data related to police arrests or use of force incidents. 
# 
# >#### What are Shapefiles?
# Without going into too much detail, Shapefiles contain geospatial information (e.g. the shape of the boundaries of a U.S. County within the context of some coordinate system stored in a .shp file) and attributes about the geographical entity (e.g. population of the U.S. County stored in a .dbf file). 
# 
# Now, our job is to somehow **combine** or **relate** these three sources of data. How can we do that? By using any **geospatial context** given to us in the datasets. By geospatial context, I mean some geographical information that is shared between the three sets of data. A simple (unrelated) example would be a zip code. Let's say each row of the Census data contained a zip code that related the collected data on poverty, race, etc., to a geographical location within a city. Now lets imagine that each row of the police arrest data also used a zip code to "geotag" an incident. We could then relate the police data to the census data using the zip code.
# 
# Unfortunately, reality is more complicated. The census (ACS) data is broken down into [Census tracts](https://transition.fcc.gov/form477/Geo/more_about_census_tracts.pdf). Census tracts have numerical IDs similar to zip codes or FIPS codes. These numerical IDs are 11 digits long. The first 5 digits are actually just FIPS codes, containing information about the U.S. State and County. The last 6 digits are the census tract. An example from `Dept_11-00091/11-00091_ACS_data/ACS_16_5YR_S1701_with_ann.csv` is under the `GEO.id2` column, 25027700100, which corresponds to Census Tract 7001, Worcester County, Massachusetts.
# 
# While the census data is relatively clean and standardized, the police data is a bit of a mess. Some Shapefiles are missing the projection information (.prj file), and borders of the police districts obviously don't align perfectly with the borders of the census tracts. We're forced to make do with what we have. Normally, the "prepped" CSV police arrest data is geotagged with either latitude and longitude or a street address. Sometimes we'll have to infer the projection of the police district Shapefiles. Normally, they contain some information to uniquely identify districts, such as a name or a numerical ID.
# 
# One important thing to note: we aren't given Shapefiles for the Census tracts, which will be necessary if we want to relate the police data (shapefiles and lat-lon points) to the census data. Luckily, they can be downloaded [here](https://www.census.gov/geo/maps-data/data/cbf/cbf_tracts.html), for each state.
# 
# ## 3. So, What's the plan?
# The plan always becomes more obvious after we've played with the data a bit, but to summarize upfront, we want to relate each row (i.e. arrest or incident) in the "prepped" police data to a corresponding Police district from the Shapefiles given, and a corresponding Census tract from the above linked Shapefiles. We can accomplish this using some form of geopandas `sjoin`,  `contains`, or `within` command. Once we relate the police data to a Census tract, we can relate it to the socio-economic/race data from the ACS surveys. Now we can dive into the code...
# 
# 

# # 4. Code <a id='start_of_code'></a>

# First, import some useful libraries...

# In[ ]:


import numpy as np
import geopandas as gpd
from geopandas.tools import sjoin
from matplotlib import pyplot as plt
import pandas as pd
from shapely.geometry import Point
import os


# To simplify things, first we'll take a look at just `Dept_37-00027`, since it contains a "prepped" CSV file.

# In[ ]:


dept_of_interest = "Dept_37-00027"
dept_folder = "../input/data-science-for-good/cpe-data/" + dept_of_interest + "/"

census_data_folder, police_shp_folder, police_csv = os.listdir(dept_folder)

# First we'll look at the Police data
for file in os.listdir(dept_folder+police_shp_folder):
    if ".shp" in file:
        shp_file = file

# Use Geopandas to read the Shapefile
police_shp_gdf = gpd.read_file(dept_folder+police_shp_folder+'/'+shp_file)

# Use Pandas to read the "prepped" CSV, dropping the first row, which is just more headers
police_arrest_df = pd.read_csv(dept_folder+police_csv).iloc[1:].reset_index(drop=True)


# Since I know you're dying to see some data:

# In[ ]:


police_shp_gdf.head()


# In[ ]:


police_arrest_df.head()


# The "prepped" CSV for `Dept_37-00027` (`police_arrest_df` above) contains both Latitude and Longitude, as well as Y.COORDINATE and Y_COORDINATE.1 (the next row defines these as X-Coordinate and Y-Coordinate, respectively). Of course there are no units, or context given for X-Coordinate and Y-Coordinate. Unfortunately, X-Coordinate and Y-Coordinate is given more often than Latitude and Longitude. While eventually we should try to map these coordinates to the corresponding Latitude and Longitude, or use the addresses to "[geocode](https://pypi.org/project/geopy/)" with lat and lon, for simplicity we'll just drop all rows that don't contain a Latitude or Longitude. We also have to be careful that our Lat and Lon are stored as floats, not strings (hence the `.astype` call below)

# In[ ]:


latlon_exists_index = police_arrest_df[['LOCATION_LATITUDE','LOCATION_LONGITUDE']].dropna().index

# Only use subset of data with existing Lat and Lon, to avoid Geocoding addresses or
# "guessing" at the meaning of Y_COORDINATE and Y_COORDINATE.1
police_arrest_df = police_arrest_df.iloc[latlon_exists_index].reset_index(drop=True)
police_arrest_df['LOCATION_LATITUDE'] = (police_arrest_df['LOCATION_LATITUDE']
                                         .astype('float'))
police_arrest_df['LOCATION_LONGITUDE'] = (police_arrest_df['LOCATION_LONGITUDE']
                                         .astype('float'))


# Right now, we're still using pure `pandas`. But if we want to take advantage of tools within `geopandas`, we'll need to convert this Lat-Lon information into Shapefile-feature-type `Points`, using `shapely.geometry`. Note that shapely uses Longitude-first convention when defining `Points`. We'll then convert the pandas `DataFrame` into a geopandas `GeoDataFrame`. We'll also assume the Lat-Lon is using the EPSG:4326 coordinate system (projection?), which we'll initialize using the `crs` attribute of the `GeoDataFrame`.

# In[ ]:


# important to check if order in Shapefile is Point(Longitude,Latitude)
police_arrest_df['geometry'] = (police_arrest_df
                                .apply(lambda x: Point(x['LOCATION_LONGITUDE'],
                                                       x['LOCATION_LATITUDE']), 
                                       axis=1))
police_arrest_gdf = gpd.GeoDataFrame(police_arrest_df, geometry='geometry')
police_arrest_gdf.crs = {'init' :'epsg:4326'}


# Now, we'd like to merge the above `police_arrest_gdf` with the police district Shapefiles. At this point, you should go upvote Chris Crawford's [kernel](https://www.kaggle.com/crawford/another-world-famous-starter-kernel-by-chris), and then comeback and upvote mine ;). Chris discovered that the `cpe-data.zip\Dept_37-00027\37-00027_Shapefiles` directory is missing the very important `.prj` file, which tells us which coordinate system projection the Shapefile is using.
# 
# Thank you Chris for finding a suitable projection. In the future, it would be cool to find someway to automate this process, using contextual clues within the Shapefile, such as significant digits of the Polygon coordinates (to decide if they're in feet or degrees) or other location information. One thing I wanted to point out is that I don't think we need to define our own `.prj` file as Chris does. I think we can stay within geopandas, and `init` our projection. We can then use `.to_crs` to convert to EPSG:4326, so that all of our data uses the same coordinate system projection.

# In[ ]:


police_shp_gdf.crs = {'init' :'esri:102739'}
police_shp_gdf = police_shp_gdf.to_crs(epsg='4326')


# We now have our two **cleaned and standardized** police`GeoDataFrames`:

# In[ ]:


police_shp_gdf.head()


# In[ ]:


police_arrest_gdf.head()


# And we can use `matplotlib` and `geopandas` to visualize:

# In[ ]:


fig1,ax1 = plt.subplots()
police_shp_gdf.plot(ax=ax1,column='SECTOR')
police_arrest_gdf.plot(ax=ax1,marker='.',color='k',markersize=4)
fig1.set_size_inches(7,7)


# Now we turn our attention back to the **census data**:

# In[ ]:


for folder in os.listdir(dept_folder+census_data_folder):
    if 'poverty' in folder:
        poverty_folder = folder


# In[ ]:


poverty_acs_file_meta, poverty_acs_file_ann = os.listdir(dept_folder+
                                                   census_data_folder+'/'+
                                                   poverty_folder)


# As with the police data, we use `pandas` to read in CSV's, and `geopandas` to read in the Census Tract Shapefile that we downloaded [here](https://www.census.gov/geo/maps-data/data/cbf/cbf_tracts.html). We can then merge the two datasets using the 11-digit Census Tract ID column that's contained within both datasets. Care must be taken that they use the same column header (hence the `.rename` command below). Finally we also make sure that they are using the same EPSG:4326 projection as the police data, using the `.to_crs` command

# In[ ]:


# Same idea as above, use pandas for CSV's and geopandas for Shapefiles
census_poverty_df = pd.read_csv(dept_folder+
                             census_data_folder+'/'+
                             poverty_folder+'/'+
                             poverty_acs_file_ann)

census_poverty_df = census_poverty_df.iloc[1:].reset_index(drop=True)

# Rename Census Tract ID column in ACS Poverty CSV to align with Census Tract Shapefile
census_poverty_df = census_poverty_df.rename(columns={'GEO.id2':'GEOID'})

census_tracts_gdf = gpd.read_file("../input/cb-2017-48-tract-500k/"+
                                  "cb_2017_48_tract_500k.shp")

# Merge Census Tract GeoDataFrame (from Shapefile) with ACS Poverty DataFrame
# using the 'GEOID', or Census Tract 11-digit numerical ID.
census_merged_gdf = census_tracts_gdf.merge(census_poverty_df, on = 'GEOID')

# Make sure everything is using EPSG:4326
census_merged_gdf = census_merged_gdf.to_crs(epsg='4326')


# Finally we can overlay the Census Tract map on top of the police data map.

# <a id='punchline'></a>
# Not the most beautiful map, but scroll Down for the punchline...

# In[ ]:


fig2,ax2 = plt.subplots()
police_shp_gdf.plot(ax=ax2,column='NAME')
police_arrest_gdf.plot(ax=ax2,marker='.',color='k',markersize=5)
census_merged_gdf.plot(ax=ax2,color='#74b9ff',alpha=.4,edgecolor='white')
fig2.set_size_inches(10,10)


# ## The Punchline:
# Now we can see the problem more clearly. Each black dot above contains information about an police arrest or use of force incident. Each transparent blue Census Tract above contains socio-economic information. And each of the multicolored shapes on the bottom layer contains information about a unique police district. The challenge is to combine all of this information in a meaningful way. Just overlaying the maps doesn't tell us much. We need to relate the attributes in each of the `GeoDataFrames`, and then we can start doing the real work: developing a useful/interesting story for CPE to share with these departments.
# 
# For each arrest `Point`, we need to assign a police district and census tract using geopandas `sjoin` or `within`/`contains` commands. ~~I've included an example of geopandas `sjoin` below, to give an example of what I'm talking about,  but unfortunately this is where I must end for now.~~ Unfortunately, as [Lilly](https://www.kaggle.com/lillyraud) mentioned in a comment bellow,  there seems to be a problem with geopandas `sjoin`, specifically the `rtree` dependency, in the current conda/Python docker version that Kaggle is using. If you have any insight into how to solve this issue, please post a comment below.
# 
# ## Merging all of the data...
# So instead we'll use Shapely's `contains` command to see if the `Polygons` of the Police District and Census Tract shapefiles "contain" any of the `Points` in the Police Arrest shapefiles. Note that each cell or row of the `geometry` column (also known as [GeoSeries](http://geopandas.org/data_structures.html#geoseries))of a geopandas GeoDataFrame is actually just a shapely object. Shapely objects have a `contains` [method](http://toblerity.org/shapely/manual.html#object.contains). The `contains` method returns True if the `Point` lies within the `Polygon`, and False if it does not. 
# 
# To reiterate, we want to check each arrest `Point`, and assign a police district and census tract. Therefore we can call the `contains` method for each Polygon, checking each arrest `Point`. If the `Point` lies within the `Polygon`, the method returns True, and we assign the name of the Police District or Census Tract as a new column in the Police Arrest GeoDataFrame. If you're not familiar with the pandas `apply` [method](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.apply.html), think of it as an alternative to running a for loop on a pandas DataFrame or Series. In the code below, I loop over all the `Polygons` and `apply` the `contains` method to all of the rows of the `Points` to avoid a nested for loop.  This allows me to create a "boolean mask" to select only the True values from the Police Arrest `Points`, and assign the name of the Police District or Census Tract to the Police Arrest.  With this many `Points` and `Polygons`, I'm worried this might take a while so I'm importing the `time` module to see how long it takes to run.

# In[ ]:





# In[ ]:




