#!/usr/bin/env python
# coding: utf-8

# # CPE Data Science for Good Challenge
# ### Integrating data with different geographic units to promote policing equity.
# 
# This notebook offers solutions to the Kaggle challenge from the Center for Policing Equity (CPE). CPE is building a National Justice Database to track police activity in participating cities. They would like to integrate this data with geographic boundary files of the local police precincts. They would also like to integrate policing data with social and economic data about the local community that is collected at the census-tract level. The key aspects of the challenge are:
# 
# >Our biggest challenge is automating the combination of police data, census-level data, and other socioeconomic factors. Shapefiles are unusual and messy -- which makes it difficult to, for instance, generate maps of police behavior with precinct boundary layers mixed with census layers. Police incident data are also very difficult to normalize and standardize across departments since there are no federal standards for data collection..
# 
# >Main Prize Track submissions will be judged based on the following general criteria:
# 
# >Performance - How well does the solution combine shapefiles and census data? How much manual effort is needed? CPE will not be able to live-test every submission, so a strong entry will be able to automate using shape files with different projections and clearly articulate why it is effective at tackling the problem.
# 
# >Accuracy - Does the solution provide reliable and accurate analysis? How well does it match census-level demographics to police deployment areas?
# 
# >Approachability - The best solutions should use best coding practices and have useful comments. Plots and graphs should should be self-explanatory. CPE might use your work to explain to stakeholders where to take action,so the results of your solution should be developed for an audience of law enforcement professionals and public officials.
# 
# The notebook is organized as follows:
# 
# 1. Integrating census tract data with Law Enforcement Agency boundary files by overlaying LEA boundaries with census tract boundaries.
# 2. Automatically downloading census data through the Census Bureau API.
# 3. Integrating police activity data by generating geographic data files from raw latitude/longitude data, and then using point-in-polygon overlay to generate precinct-level data on police activity.
# 4. Demonstrating how integrated data from police activity and the US Census can be used to generate metrics related to police bias.
# 
# CPE has provided LEA shape files, police activity data, and census tract data files for 12 partner cities in the US. The type of police activity data provided varies across the cities, as do the time periods that are covered. Six cities provide 'Use of Force' data, although two seem to contain only 35 and 79 instances and so are probably incomplete. Three cities provide 'Incidents' data, one provides 'Field Interviews', another 'Arrests', and another 'Vehicle Stops'. There are a variety of issues with these files, for example the shapefile for the LEA boundaries in Austin, TX is missing projection information, the LEA shapefile for Dallas, TX contains only a subset of precincts, and the shapefile for Charlotte, NC contains point data instead of polygons. There are also undoubtedly some issues with the data files that will require cleaning and transformation. The focus here will be less on data cleaning and more on the spatial transformations that can be used to integrate data at different geographic levels. To do this I will demonstrate with the data for San Francisco, CA as an example case.
# 
# The concepts presented here are intended to be generalizable to other locations or datasets beyond those used here. This notebook uses Python 3, including common libraries such as Numpy, Pandas, Matplotlib, etc. It also uses the libraries Geopandas for spatial operations and CensusData for accessing the Census Bureau API. There are many tools or sets of tools that could be used depending on the desired application and scale.

# ## 1. Integrating census tract data with law enforcement agency boundary files.

# In[ ]:


# Set up environment
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import geopandas as gpd

# For Windows: To address problem with current windows build of GeoPandas
# https://github.com/geopandas/geopandas/issues/830
#import os
#os.environ["PROJ_LIB"] = "C:\ProgramData\Anaconda3\Library\share" #windows


# In[ ]:


# Find file locations on Kaggle
import os
os.listdir("../input/")


# In[ ]:


#Read in law enforcement agency (LEA) shapefile and check projection
#lea_shape = gpd.read_file("../input/data-science-for-good/cpe-data/Dept_49-00081/49-00081_Shapefiles/SFPD_geo_export_8a7e3c3b-8d43-4359-bebe-b7ef6a6e21f3.shp")
lea_shape = gpd.read_file("../input/data-science-for-good/cpe-data/Dept_49-00081/49-00081_Shapefiles/SFPD_geo_export_8a7e3c3b-8d43-4359-bebe-b7ef6a6e21f3.shp")
print(lea_shape.crs)
print(lea_shape.shape)
lea_shape.head()


# #### The LEA shapefile for San Francisco has 10 rows (identified as police districts) and 6 columns. The projection is espg: 4326.

# In[ ]:


# Read in shape file for census tracts for the state. 
# This file can also be accessed directly from the Census Bureau via URL. 
# Tract maps for counties are also available, but some police departments might span counties (e.g., NYC).
state_shape = gpd.read_file("../input/ca-tract-shapefile/cb_2017_06_tract_500k.shp")
#state_shape = gpd.read_file("./cb_2017_06_tract_500k/cb_2017_06_tract_500k.shp")

print(state_shape.crs)
print(state_shape.shape)
state_shape.head()


# #### The tract shapefile for California has 8041 rows and 10 columns. The unique tract identifier is the combination of the state FIPS code (STATEFP), the county FIPS code (COUNTYFP), and the tract (TRACTCE or NAME). Tract numbers are reused across counties, but since we have census data for just one county we can use the tract identifier alone. 
# 
# #### The projection is espg: 4269, which is not the same as the LEA shapefile. In order to plot these correctly on the same map or to perform spatial operations correctly, we need to transform one of these boundary files so that the projection will be the same. I will keep the projection from the census files, since we may encounter this projection again in other census files.

# In[ ]:


# Convert LEA geometry to same projection as Census files
lea_shape = lea_shape.to_crs({'init': 'epsg:4269'}) 


# In[ ]:


# Quick plot shows the 10 police districts
lea_shape.plot(column = 'district')


# In[ ]:


# Plot both LEA districts and census tracts together

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
fig.set_size_inches(10, 10, forward=True)

ax.set_aspect('equal')
ax.set_xlim(-122.52, -122.36)
ax.set_ylim(37.7, 37.84)
ax.set_title("Overlay of San Francisco Police Districts with Census Tracts")

lea_shape.plot(ax=ax, color='white', edgecolor='black')
state_shape.plot(ax=ax, color='white', edgecolor='blue', alpha=.1)
plt.show();


# #### Above we see that in general there are many census tracts per LEA district. Sometimes the boundaries coincide, but some tracts are split between two police divisions. Next we will create an intersection using these shapes that will allow us to convert demographic and social data at the census tract level to the LEA district level. 
# 
# #### Kaggle does not currently support the GeoPandas overlay, so I have uploaded the processed file for use on this platform.

# In[ ]:


# Create new Geodataframe that contains the intersections
# Will not work on Kaggle
#intersect = gpd.overlay(state_shape, lea_shape, how='intersection')


# In[ ]:


# for Kaggle
# Export intersect dataframe so it can be used on Kaggle platform
#intersect.to_pickle("./Dept_49-00081/Dept_49-00081_intersect.pkl")

# Read in pickled dataframe with the results of the overlay operation
intersect = pd.read_pickle("../input/intersection/Dept_49-00081_intersect.pkl")


# In[ ]:


intersect.TRACTCE.value_counts().head()


# #### Notice that because some tracts intersect with multiple LEA districts, they have been split into smaller polygons.

# In[ ]:


# Select tract that has been split into 4 parts
tract_new = intersect[intersect['TRACTCE'] == "016801"]

# Show shape of original tract
tract_old = state_shape[state_shape['TRACTCE'] == "016801"]

figure1, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
figure1.set_size_inches(10, 4, forward=True)
ax1.set_title('Original census tract 016801')
ax1.set_yticks([0.001, 0.003])

# Show tract split by intersection
tract_old.plot(ax=ax1, color='white', edgecolor='black')
tract_new.plot(ax=ax2, color='white', edgecolor='black')
ax2.set_title("Intersection of census tract with LEA districts")
plt.show()


# #### The above plot shows how this census tract has been split into 4 parts by the intersection. Next we will use these new polygons to transform census tract-level data to LEA district-level. The first step is to calculate some areas.
# 
# 

# In[ ]:


# Get area of intersection polygons 
intersect['inter_area'] = intersect.area
# And drop unneeded columns
#intersect = intersect.drop(columns = ['ALAND', 'AWATER', 'shape_area', 'shape_le_1', 'shape_leng'])


# In[ ]:


# Get area of each tract by summing its polygons
tract_area = intersect.groupby("TRACTCE").agg({'inter_area': 'sum'}).rename(columns={'inter_area': 'tract_area'})
tract_area.head(2)


# In[ ]:


# Get area of each LEA by summing its polygons
LEA_area = intersect.groupby("company").agg({'inter_area': 'sum'}).rename(columns={'inter_area': 'LEA_area'})

# Merge intersection dataframe with tract and LEA area data
intersect_area = intersect.merge(tract_area, how='left', on='TRACTCE').merge(LEA_area, how='left', on='company')

# Calculate polygon percent of tract and polygon percent of LEA district
intersect_area['prop_of_tract'] = intersect_area.inter_area / intersect_area.tract_area
intersect_area['prop_of_LEA'] = intersect_area.inter_area / intersect_area.LEA_area
intersect_area.info()


# #### For each intersection polygon in this dataframe, we now also know what proportion of land area the polygon contributes to the tract it falls within and the LEA district it falls within. We can apply these proportions to data at the census tract level and then aggregate the polygons to the LEA level to obtain LEA estimates of census measures. 
# 
# #### To demonstrate, we have to get some census tract-level data. For now, we will just grab the total population estimate. Later we will look at census data more in-depth.

# In[ ]:


# Read in ACS 2015 5yr DP05 file from Kaggle
file = '../input/data-science-for-good/cpe-data/Dept_49-00081/49-00081_ACS_data/49-00081_ACS_race-sex-age/ACS_15_5YR_DP05_with_ann.csv'
dp05 = pd.read_csv(file, skiprows=[1])[['HC01_VC03', 'GEO.id', 'GEO.id2', 'GEO.display-label']].rename(columns={'HC01_VC03': 'totpop'})

# Extract tract number from text string
split1 = dp05['GEO.display-label'].str.split(',', expand=True)
split2 = split1[0].str.split(' ', expand=True)
dp05['tract_num'] = split2[2]


# In[ ]:


dp05.head()


# In[ ]:


# Merge on NAME (instead of TRACTCE) because it is an easier match to the census data
intersect_data = intersect_area.merge(dp05, how='left', left_on='NAME', right_on='tract_num')


# In[ ]:


intersect_data[intersect_data['tract_num'].isna()]


# #### This merge shows us that our intersection picked up a few census tracts that are not in the census data. Notice that each of these polygons make up a tiny fraction of the LEA district they intersected with. These tracts have been picked up in error due to small misalignments between the census tract and LEA district maps. We can delete them now. Alternatively, we could have eliminated these earlier in the process by only taking census data from San Francisco county. But not all LEA boundaries coincide with county boundaries, so it is important to make that decision on a case by case basis.

# In[ ]:


# Get rid of LEA overlap polygons outside of San Francisco County
intersect_data = intersect_data[intersect_data['tract_num'].notna()]

# Estimate population in each intersection polygon
# Assumes that each polygon has the same proportion of population as it has of land area.
intersect_data['poly_pop'] = intersect_data.totpop * intersect_data.prop_of_tract

# Aggregate intersection polygons to the LEA district level
LEA_pop = intersect_data.groupby("company").agg({'poly_pop': 'sum'}).rename(columns={'poly_pop': 'totpop'})
#LEA_pop.info()


# In[ ]:


# Check to see if LEA data has same total as original census data.
print("San Francisco population from original census data: ", dp05['totpop'].sum() )
print("San Francisco population from new LEA data: ", LEA_pop['totpop'].sum() )


# In[ ]:


# Merge LEA-level census data to LEA geo data for mapping
lea_shape_data = lea_shape.merge(LEA_pop, how='left', on='company')


# In[ ]:


# Make choropleth map of population in LEA districts
fig, ax = plt.subplots()
fig.set_size_inches(11, 7, forward=True)

ax.set_aspect('equal')
ax.axis('off')
ax.set_title("Population estimates for San Francisco Police Districts, ACS 5yr 2015", fontsize=15)

lea_shape_data.plot(ax=ax, column='totpop', cmap='BuPu', edgecolor='black')

# Create colorbar as a legend
vmin, vmax = np.min(lea_shape_data['totpop']),  np.max(lea_shape_data['totpop'])
sm = plt.cm.ScalarMappable(cmap='BuPu', norm=plt.Normalize(vmin=vmin, vmax=vmax))
# empty array for the data range
sm._A = []
# add the colorbar to the figure
cbar = fig.colorbar(sm)

plt.show();


# #### Above is a choropleth map of estimates of the ACS 5yr 2015 population for the 10 police districts in San Francisco. This technique can be used to generate estimates for many different census counts such as breakdowns by race, sex, age, poverty status, educational attainment, employment, etc. It is important to note that estimates produced in this way are subject to the assumption that the population in question is evenly distributed throughout the census tract.

# ## 2. Census Data

# 
# Outside of the decennial census, the primary source for census data at the geographic level of the census tract or lower is the 5-year American Community Survey data. The 5-year ACS is a rich source of data. Tables are available containing many detailed breakdowns, and the full range of long-form census questions are included. The tradeoff, however, is that this data represents an overall 5-year period. So, for example, the 2017 ACS 5-year data is reflective of the period 2013-2017. Updated 5-year tables are released each year, but they are not intended to be used as moving averages.  The 5-year window allows the Census Bureau to protect individual privacy while managing a large scale survey research process. The names of the data files provided for this challenge indicate that the data is from the 2015 5yr ACS release, which covers the years 2011-2015.
# 
# The files provided by CPE are ACS 5yr 2015 files that have been downloaded from the American Fact Finder website. Most are from Data Profile tables, which provide many of the most popular demographic measures. It can be convenient to obtain census data in this way. But the US Census Bureau offers an API for more direct access, and using this process can have some advantages:
# 
# - When files are manually downloaded, the requested columns are automatically assigned names, which are documented in the associated metadata file. These names are reused for each downloaded file. Accessing data through the API allows the use of unique names which provides for better documentation of the download process.
# - The API allows users to access only the columns that are required, rather than downloading entire tables.
# - Code can be reused by passing parameters to get data for different time periods or different locations.
# 
# Some disadvantages of accessing census data through the API:
# - Requires lookup of census table and column identifiers, which can be tedious.
# - Writing code can be more time consuming than using the interactive download page, especially if only a few files are needed.
# 
# Directly accessing the Census API is not possible on the Kaggle platform, so for this section I have again provided the file that would result from running the code in the hidden cells.

# In[ ]:


# One way to access the Census Bureau API is through the Python package CensusData
# https://github.com/jtleider/censusdata
#import censusdata
#import pandas as pd
#from sodapy import Socrata

# The search method can be used to identify tables containing keyword of interest
# Just an example of how to search for tables. This will produce long output.
#censusdata.search('acs1', '2015', 'label', 'housing')

# The printtable method can get field names and descriptions from a specific table. 
# Also produces long output. Running the code below will retreive documentation of the DP05 table
# censusdata.printtable(censusdata.censustable('acs5', '2015', 'DP05'))

# Method to get list of available geographies
#censusdata.geographies(censusdata.censusgeo([('state', '06'), ('county', '075'), ('tract', '*')]), 'acs5', 2015)


# In[ ]:


# Will not work on Kaggle
# Define function to retrieve census data
def get_census(file, year, geography, census_vars):
    census_raw = censusdata.download(file, year, geography, census_vars, tabletype='profile') 
    census_raw.reset_index(inplace = True)
    census_raw['tract_string'] = census_raw['index'].astype(str)
    # Extract tract number from text string
    split1 = census_raw['tract_string'].str.split(',', expand=True)
    split2 = split1[0].str.split(' ', expand=True)
    census_raw['tract_num'] = split2[2]
    census_raw = census_raw.drop(columns=['index'])
    return census_raw


# In[ ]:


# Will not work on Kaggle
# Define parameters for census download
year = 2015
file = 'acs5'
geography = censusdata.censusgeo([('state', '06'), ('county', '075'), ('tract', '*')])
census_vars = ['DP05_0001E', 
'DP05_0002E', 
'DP05_0003E', 
'DP05_0004E', 
'DP05_0005E', 
'DP05_0006E', 
'DP05_0007E', 
'DP05_0008E', 
'DP05_0009E', 
'DP05_0010E', 
'DP05_0011E', 
'DP05_0012E', 
'DP05_0013E', 
'DP05_0014E', 
'DP05_0015E', 
'DP05_0016E', 
'DP05_0018E', 
'DP05_0019E', 
'DP05_0020E', 
'DP05_0021E', 
'DP05_0028E', 
'DP05_0030E', 
'DP05_0032E', 
'DP05_0033E', 
'DP05_0034E', 
'DP05_0039E', 
'DP05_0047E', 
'DP05_0052E', 
'DP05_0066E',  
'DP05_0081E']


# In[ ]:


# Will not work on Kaggle
#Download requested data from census bureau
C = get_census(file, year, geography, census_vars)


# In[ ]:


# For Kaggle
C = pd.read_pickle("../input/census-extract/census_extract.pkl")


# In[ ]:


# Rename columns
C = C.rename(columns={'DP05_0001E' : 'totpop', 
'DP05_0002E' : 'sex_male', 
'DP05_0003E' : 'sex_female', 
'DP05_0004E' : 'age_under_5', 
'DP05_0005E' : 'age_5_9', 
'DP05_0006E' : 'age_10_14', 
'DP05_0007E' : 'age_15_19', 
'DP05_0008E' : 'age_20_24', 
'DP05_0009E' : 'age_25_34', 
'DP05_0010E' : 'age_35_44', 
'DP05_0011E' : 'age_45_54', 
'DP05_0012E' : 'age_55_59', 
'DP05_0013E' : 'age_60_64', 
'DP05_0014E' : 'age_65_74', 
'DP05_0015E' : 'age_75_84', 
'DP05_0016E' : 'age_85_over', 
'DP05_0018E' : 'age_18_over', 
'DP05_0019E' : 'age_21_over', 
'DP05_0020E' : 'age_62_over', 
'DP05_0021E' : 'age_65_over', 
'DP05_0028E' : 'race_total', 
'DP05_0030E' : 'race_multi', 
'DP05_0032E' : 'race_white', 
'DP05_0033E' : 'race_black', 
'DP05_0034E' : 'race_native_amer', 
'DP05_0039E' : 'race_asian', 
'DP05_0047E' : 'race_pac_island', 
'DP05_0052E' : 'race_other', 
'DP05_0066E' : 'hispanic', 
'DP05_0081E' : 'housing_units'})

#C.info()


# In[ ]:


# Merge on intersection polygons with census data on NAME & tract_num
intersect_data = intersect_area.merge(C, how='left', left_on='NAME', right_on='tract_num')

#Sum polygon counts to LEA level
grouped = intersect_data.groupby("company")
LEA_census_data = grouped.sum().drop(columns=['inter_area', 'tract_area', 'LEA_area', 'prop_of_tract', 'prop_of_LEA'])


# In[ ]:


# Estimate population metric in each intersection polygon
def calc_LEA_percent(df, vars):
    for var in vars:
        df[var+"_p"] = 100 * df[var] / df['totpop']


# In[ ]:


varlist = list(LEA_census_data.columns)
calc_LEA_percent(LEA_census_data, varlist)
#LEA_census_data.info()


# In[ ]:


# Merge LEA-level census data to LEA geo data for mapping & analysis
LEA_shape_data = lea_shape.merge(LEA_census_data, how='left', on='company').drop(columns='totpop_p')
#LEA_shape_data.info()


# #### Now we have a geo data file that includes LEA boundaries and census data that has been converted from tract level to LEA district level. We can examine some of the demographic features of the police districts.

# In[ ]:


# Make choropleth map of population in LEA districts
fig, ax = plt.subplots()
fig.set_size_inches(11, 7, forward=True)

ax.set_aspect('equal')
ax.axis('off')
ax.set_title("Percent Asian Population for San Francisco Police Districts, ACS 5yr 2015", fontsize=15)

LEA_shape_data.plot(ax=ax, column='race_asian_p', cmap='BuPu', edgecolor='black')

# Create colorbar as a legend
vmin, vmax = np.min(LEA_shape_data['race_asian_p']),  np.max(LEA_shape_data['race_asian_p'])
sm = plt.cm.ScalarMappable(cmap='BuPu', norm=plt.Normalize(vmin=vmin, vmax=vmax))
# empty array for the data range
sm._A = []
# add the colorbar to the figure
cbar = fig.colorbar(sm)

plt.show();


# In[ ]:


varlist = ['race_asian_p','race_black_p', 'race_multi_p', 'race_native_amer_p', 'race_other_p', 'race_pac_island_p', 'race_white_p', 'hispanic_p']
var_labels = ['Asian','Black', 'Multi-race', 'Native American', 'Other', 'Pacific Island', 'White', 'Hispanic']

fig = plt.figure()
fig.set_size_inches(16, 20, forward=True)
fig.suptitle('Race/Ethnic Composition of LEA Districts', fontsize=16)

num=1

for var in varlist:
    plt.subplot(4, 2, num)
    
    # Choose the height of the bars
    height = LEA_shape_data[var]
 
    # Choose the names of the bars
    bars = list(LEA_shape_data['district'].unique())
    y_pos = np.arange(len(bars))
 
    # Create bars
    plt.bar(y_pos, height)
 
    # Create names on the x-axis
    plt.xticks(y_pos, bars, rotation=45, fontsize=8)
    plt.subplots_adjust(top=0.9, bottom=.1, hspace=.3)
    plt.tight_layout
    plt.title(var_labels[num-1])
    plt.ylabel("Percent")
 
    num+=1

# Show graphic
plt.show()


# ## 3. Police Activity Data
# 
# The data provided for San Francisco are incidents for 2012 through May of 2015. The next step is to explore this policing data and convert the raw data to geo data.

# In[ ]:


#Import LEA data file
file = '../input/data-science-for-good/cpe-data/Dept_49-00081/49-00081_Incident-Reports_2012_to_May_2015.csv'
LEA_incidents_raw = pd.read_csv(file, skiprows=[1])
LEA_incidents_raw.info()


# In[ ]:


# Show indicent reasons
LEA_incidents_raw.INCIDENT_REASON.value_counts()


# In[ ]:


# Show most common incident dispositions
LEA_incidents_raw.DISPOSITION.value_counts().head(20)


# #### The policing data for San Francisco contains 394,235 records and 11 columns. There is no missing data and on preliminary inspection the data file appears to be very clean and tidy. It is also a rather slim dataset, as it does not give us any descriptive information about the officers, suspects, or victims involved. Therefore we will not be able to investigate police bias based on this dataset alone and we will not be able to investigate any impacts of the demographics (race, sex age, etc.) of the persons involved. 
# 
# #### That leaves us with examining patterns of police activity and how they may differ across communities in San Francisco. This could be done at either the census tract or the LEA district level. The choice depends on how the information will be used. For this demonstration we will stick with using the LEA boundaries as the aggregation level for both census data and policing data. Next we will convert the police activity data to a geodataframe.

# In[ ]:


from shapely.geometry import Point

# Create tuple of lat/long
LEA_incidents_raw['Coordinates'] = list(zip(LEA_incidents_raw.LOCATION_LONGITUDE, LEA_incidents_raw.LOCATION_LATITUDE))
# Convert to point
LEA_incidents_raw['Coordinates'] = LEA_incidents_raw['Coordinates'].apply(Point)
# Convert to geodataframe
LEA_incidents_shape = gpd.GeoDataFrame(LEA_incidents_raw, geometry='Coordinates')
# Set point geometry to same projection as Census files
LEA_incidents_shape.crs = {'init': 'epsg:4269'}


# In[ ]:


LEA_incidents_shape.head()


# In[ ]:


# Plot LEA districts, census tracts, and police incidents together

embez = LEA_incidents_shape[LEA_incidents_shape['INCIDENT_REASON']=="EMBEZZLEMENT"]

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
fig.set_size_inches(10, 10, forward=True)

# set aspect to equal. This is done automatically
# when using *geopandas* plot on it's own, but not when
# working with pyplot directly.
ax.set_aspect('equal')
ax.set_xlim(-122.52, -122.36)
ax.set_ylim(37.7, 37.84)
ax.set_title("Embezzlement Incidents in San Francisco", fontsize=15)

lea_shape.plot(ax=ax, color='white', edgecolor='black')
state_shape.plot(ax=ax, color='white', edgecolor='blue', alpha=.1)
embez.plot(ax=ax, color='red', alpha=.5)
plt.show();


# #### Embezzlement incidents seem to occur more often in the central business district. What about drug offenses?

# In[ ]:


# Plot LEA districts, census tracts, and police incidents together

drug = LEA_incidents_shape[LEA_incidents_shape['INCIDENT_REASON']=="DRUG/NARCOTIC"]

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
fig.set_size_inches(10, 10, forward=True)

# set aspect to equal. This is done automatically
# when using *geopandas* plot on it's own, but not when
# working with pyplot directly.
ax.set_aspect('equal')
ax.set_xlim(-122.52, -122.36)
ax.set_ylim(37.7, 37.84)
ax.set_title("Drug/Narcotic Incidents in San Francisco", fontsize=15)

lea_shape.plot(ax=ax, color='white', edgecolor='black')
state_shape.plot(ax=ax, color='white', edgecolor='blue', alpha=.1)
drug.plot(ax=ax, color='green', alpha=.3)
plt.show();


# #### Drug related incidents are much more common than embezzlement, and are prevalent in several areas outside the CBD.
# 
# #### Since these geodataframes all have the same projection they can easily be plotted together without actually merging the data. we will need to combine the files in order to generate statistics using the police incident data and the census data that we transformed to LEA districts. Geodataframes can be joined based either on attributes (matching on a shared column) or based on the spatial relationship between the two geometries. 
# 
# #### In this case we want to join the LEA boundaries with the LEA incident points so that we can assign an LEA district to each point. Then we can count the incidents in each district and generate other metrics. Because the San Francisco incidents data is so tidy, it already contains a column with the district name that we can probably use for an attribute join (assuming all the spelling lines up). But it is not reasonable to expect that all police data will be this tidy, and since this notebook is intended to be a demonstration of automating the integration of spatial data, I will use the spatial join. This type os spatial join is also sometimes called a point-in-polygon operation. As mentioned before, we could alternatively do the spatial join with incidents and census tract boundaries and perform an analysis at that level.
# 
# #### Note that this operation also is not supported by Kaggle, so I have provided the resulting data file.

# In[ ]:


#Will not work on Kaggle
#incidents_LEA_districts = gpd.sjoin(LEA_incidents_shape, lea_shape, how="inner", op='intersects')

# Read in result of spatial join
incidents_LEA_districts = pd.read_pickle("../input/spatial-join/incidents_LEA_districts.pkl")
incidents_LEA_districts.info()


# #### Notice that we have lost 131 incident records in this inner join. We could potentially identify those incident records and try to correct whatever issue prevented them from matching, but since we still have over 394K records to use for demonstration we will let those go for now.
# 
# #### Next we will aggregate this data to the LEA district level and then join it to the LEA boundaries. There are many ways this can be done, depending on the ultimate use of the data. We could count up all the incidents in each LEA district, or we could subset them by the Incident Reason and provide separate counts. We could could also aggregate by date or time or incident disposition or any specific combinations of these.
# 
# #### To keep it simple we will aggregate the data by LEA district, but we will only look at embezzlement and drug/narcotic incidents and we will aggregate them separately. We will also look at the incident disposition in these cases. Since the most common disposition is 'NONE', we will simplify this variable into a binary indicator of whether any action was taken or not.

# In[ ]:


incidents_LEA_districts['Embezzlements'] = incidents_LEA_districts['INCIDENT_REASON']=="EMBEZZLEMENT"
incidents_LEA_districts['Drugs'] = incidents_LEA_districts['INCIDENT_REASON']=="DRUG/NARCOTIC"
incidents_LEA_districts['Actions'] = incidents_LEA_districts['DISPOSITION']=="NONE" 
incidents_LEA_districts['EmbActions'] = incidents_LEA_districts.Embezzlements & incidents_LEA_districts.Actions
incidents_LEA_districts['DrugActions'] = incidents_LEA_districts.Drugs & incidents_LEA_districts.Actions


# In[ ]:


incidents_LEA_districts.head()


# In[ ]:


#Sum incident records to LEA level
grouped = incidents_LEA_districts.groupby('district')
incidents_grouped = grouped.sum().drop(columns=['index_right', 'shape_area', 'shape_le_1', 'shape_leng', 'INCIDENT_UNIQUE_IDENTIFIER', 'LOCATION_LONGITUDE', 'LOCATION_LATITUDE', 'Actions'])
#Create a few rates - remember Action means "NONE" before grouping
incidents_grouped['Emb_Action_Rate'] = 100-(100 * incidents_grouped.EmbActions/incidents_grouped.Embezzlements)
incidents_grouped['Drug_Action_Rate'] = 100-(100 * incidents_grouped.DrugActions/incidents_grouped.Drugs)


# In[ ]:


incidents_grouped.head()


# #### It's interesting that action is taken in a higher percentage of drug-related incidents than embezzlement incidents.
# 
# #### Now that we have created some police activity measures at the LEA district level, we can now join that data to the geodataframe we created that contains census data at the LEA district level

# In[ ]:


LEA_integrated = LEA_shape_data.merge(incidents_grouped, how='left', on='district')
#LEA_integrated.info()


# In[ ]:


# Make choropleth map of population in LEA districts
fig, ax = plt.subplots()
fig.set_size_inches(11, 7, forward=True)

ax.set_aspect('equal')
ax.axis('off')
ax.set_title("Percent of Drug Incidents In Which Police Action is Taken", fontsize=15)

LEA_integrated.plot(ax=ax, column='Drug_Action_Rate', cmap='Oranges', edgecolor='black')

# Create colorbar as a legend
vmin, vmax = np.min(LEA_integrated['Drug_Action_Rate']),  np.max(LEA_integrated['Drug_Action_Rate'])
sm = plt.cm.ScalarMappable(cmap='Oranges', norm=plt.Normalize(vmin=vmin, vmax=vmax))
# empty array for the data range
sm._A = []
# add the colorbar to the figure
cbar = fig.colorbar(sm)

plt.show();


# #### Police action is more often taken in downtown locations than in residential neighborhoods. Can this data also be used to investigate police bias? That is the subject of the final section of this notebook.

# ## 4. Investigating Police Bias
# 
# We have now transformed census data to the LEA district level through the use of spatial overlay and we have transformed raw lat/long point data to aggregated LEA district data using a point-in-polygon spatial join. We now have a geodataframe that contains LEA boundaries, police activity measures, and census data. We can now begin to explore what we can do with this rich dataset.
# 
# There are some limitations, though, to what we can do in this demonstration. We have downloaded only a small fraction of the data that is available via the Census Bureau API, pulling only the most basic demographics. In addition, the police activity data provided by CPE for San Francisco does not contain any data on the demographics of the officers or suspects involved in the incidents. Therefore we are not able to explore relationships between characteristics of suspects or officers on the disposition of incidents that might indicate bias on a personal level. And since we have aggregated data at the LEA district level, we are limited to looking at differences at the LEA district level.

# In[ ]:


varlist = ['race_asian_p','race_black_p', 'race_multi_p', 'race_native_amer_p', 'race_other_p', 'race_pac_island_p', 'race_white_p', 'hispanic_p']
var_labels = ['Asian Percent','Black Percent', 'Multi-race Percent', 'Native American Percent', 'Other Percent', 'Pacific Island Percent', 'White Percent', 'Hispanic Percent']

fig = plt.figure()
fig.set_size_inches(12, 12, forward=True)
fig.suptitle('Race Composition and Action Taken in Drug Incidents in LEA Districts', fontsize=16)
num=1

for var in varlist:
    plt.subplot(3, 3, num)
    plot_var =  LEA_integrated[var]
    plt.scatter(plot_var, LEA_integrated['Drug_Action_Rate'])
    plt.tight_layout
    plt.title(var_labels[num-1])
    plt.ylabel("Percent Action Taken")
 
    num+=1

# Show graphic
plt.show()


# In[ ]:


varlist = ['race_asian_p','race_black_p', 'race_multi_p', 'race_native_amer_p', 'race_other_p', 'race_pac_island_p', 'race_white_p', 'hispanic_p']
var_labels = ['Asian Percent','Black Percent', 'Multi-race Percent', 'Native American Percent', 'Other Percent', 'Pacific Island Percent', 'White Percent', 'Hispanic Percent']

fig = plt.figure()
fig.set_size_inches(15, 12, forward=True)
fig.suptitle('Race Composition and Action Taken in Drug Incidents in LEA Districts', fontsize=16)
num=1

for var in varlist:
    plt.subplot(3, 3, num)
    plot_var =  LEA_integrated[var]
    plt.scatter(plot_var, LEA_integrated['Drug_Action_Rate'])
    plt.tight_layout
    plt.title(var_labels[num-1])
    plt.ylabel("Percent Action Taken")
 
    num+=1

# Show graphic
plt.show()


# In[ ]:


varlist = ['age_15_19_p','age_20_24_p']
var_labels = ['Percent 15-19','Percent 20-24']

fig = plt.figure()
fig.set_size_inches(14, 4, forward=True)
fig.suptitle('Age Composition and Action Taken in Drug Incidents in LEA Districts', fontsize=16)
num=1

for var in varlist:
    plt.subplot(1, 2, num)
    plot_var =  LEA_integrated[var]
    plt.scatter(plot_var, LEA_integrated['Drugs'])
    plt.tight_layout
    plt.title(var_labels[num-1])
    plt.ylabel("Number of Drug Incidents")
 
    num+=1

# Show graphic
plt.show()


# #### There are no obvious relationships in the bivariate scatter plots. I hope, however, that this brief exercise has helped to demonstrate how data with varying geographies can be integrated to produce useful datasets. The potential uses for this type of data include standardized reports for local law enforcement, rich datasets for statistical analysis, flat files for export to BI, GIS, or data visualization applications.
# 
# #### Some parts of the process involved in producing the integrated data are more amenable to automation than others. The initial validation of police/LEA shapefiles will likely need human attention in some cases, although validation functions can be created to screen for certain specific requirements such as projection information. Correcting problems with shapefiles is not really possible to automate when working with files from partner institutions, but the good news is you will not have to do it often as these boundaries are not likely to change frequently.
# 
# #### The same can be said for cleaning data provided by law enforcement agencies: data cleaning can only be automated so much. If a set of data standards can be agreed upon, then it certainly possible to create validation tools that will isolate problematic records. Correcting them will require human intervention.
# 
# #### Files from the Census Bureau are usually highly reliable and well-documented. Shapefiles for tracts or other boundaries change infrequently, so they can be probably be downloaded manually. Shapefiles can also be accessed via URL. Census population data can be downloaded manually or can be accessed via an API. Writing some code to make use of the API may be worth the effort if you will want to update the data anually or if you will want to make requests for the same data for different cities as they come on board.
# 
# #### Once data is in a standardized format, it can certainly be possible to write code that will take in shape files and data files, and then produced standardized ouput for further use.

# ### Thanks for Reading!

# In[ ]:




