#!/usr/bin/env python
# coding: utf-8

# # NYPD Complaint Data Cleaning for Beginners
# ### Data updated 7 June 2018
# 
# ### Notebook created - *August 2018*
# ## All updates written out at the end.
# ### Version - 2

# ## Introduction
# 
# 
# This notebook was made as a product of data cleaning and dealing with NaNs skills practise. Many NaN values from this kernel could be eliminated in other, simpler ways without dataset damage. This is my first kernel and I'm a novice in data science, so if you find any error or you know a better solution to get same or similar results, please post in comment. What is more, I do my best with my english language writting skills, so please be understanding.

# ## Activities I am planning to perform in this kernel
# 
# 
# ### [Data exploration and cleaning:](#1)
# 1. [First things first](#2)
# 2. [Import data and explore column data types](#3)
# 3. [Drop values/columns](#4)
# 4. [Fill NaNs](#5)
# 5. [Which borough?](#6)
# 6. [Final toughts](#7)

# <a id="1"></a> <br>
# ## DATA EXPLORATION AND CLEANING

# <a id="2"></a> <br>
# ### FIRST THINGS FIRST

# In[ ]:


# Imports

# Visualisations
import matplotlib.pyplot as plt 
import matplotlib
import plotly.plotly as py
import plotly.graph_objs as go
import seaborn as sns
plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')

# Warnings
import warnings
warnings.filterwarnings('ignore')

# Data exploration
import pandas as pd

# Numerical
import numpy as np

# Spatial analysis
import geopandas as gpd   # used to read .shp file
from shapely.geometry import Point, Polygon, shape
import utm   # Need to be installed, coordinates conversion

# Regular expressions
import re

# Random
np.random.seed(11)

# Files in dataset
import os
print(os.listdir("../input"))


# In[ ]:


# Import data
crimes_raw = pd.read_csv('../input/nypd-complaint-data-current-ytd-july-2018/NYPD_Complaint_Data_Current_YTD.csv')


# <a id="3"></a> <br>
# ### IMPORT DATA AND EXPLORE COLUMN DATA TYPES

# In[ ]:


# Columns data info
crimes_raw.info()


# In[ ]:


crimes_raw.head()


# <a id="4"></a> <br>
# ### DROP VALUES/COLUMNS

# <br>As a begin, drop columns that contain duplicated or useless information for this kernel. I decided to drop *HADEVELOPT* ,  *HOUSING_PSA*,  *LOC_OF_OCCUR_DESC*,  *PD_CD* columns.
# <br>
# <br>***HADEVELOPT*** - too detailed 
# <br>***HOUSING_PSA*** - useless
# <br>***LOC_OF_OCCUR_DESC*** - useless
# <br>***PD_CD*** - too detailed, duplicated
# 
# <br> *X_COORD_CD* and *Y_COORD_CD* columns contain the similar information about location as columns *Latitude* and *Longitude* (other reference systems). In some cases I will use  *X_COORD_CD* and *Y_COORD_CD* columns' values in other *Latitude* and *Longitude*. No need for unit conversion if all values are currently given. Lat and lon coordinates are more intuitive and conveninet for me, so I'll use them.

# In[ ]:


# Columns to drop
crimes = crimes_raw.drop(['HADEVELOPT', 'HOUSING_PSA', 'LOC_OF_OCCUR_DESC', 'PD_CD'], axis = 1)

# Drop additional columns
# crimes = crimes.drop(['X_COORD_CD', 'Y_COORD_CD', 'Lat_Lon'], axis = 1)


# <a id="5"></a> <br>
# ### FILL NANS

# <br>Quite a few columns have missing values. There are several types of missing values: 
# <br>**NaN**  -  no value
# <br>**UNKNOWN** - in some cases it means value is unknown, in other it is label as unknow to prevent identifying victims
# <br>**outliers** - values inadequate for given category (e.g. *SUSP_AGE_GROUP*, value: *978*)
# <br>Almost all missing values will be filled. Several columns with incomplete data might have been dropped, but have been preserved in order to practice filling NaNs.
# <br> Details of dealing with NaNs divided into columns are below.

# <br>***CMPLN_TO_DT*** - filled missing values with *CMPLNT_FR_DT* means incident occured during one day
# <br>***CMPLNT_TO_TM*** - filled missing values with *CMPLMNT_FR_TM* means incident occured but did no last - it is a kind of simplification, detailed analysis of incidents' time is not point of thie kernel

# In[ ]:


# Fill CMPLNT_TO_DT NaNs with CMPLNT_FR_DT values.
crimes['CMPLNT_TO_DT'].fillna(crimes['CMPLNT_FR_DT'], axis = 0, inplace = True)

# Fill CMPLNT_TO_TM NaNs with CMPLNT_FR_TM values.
crimes['CMPLNT_TO_TM'].fillna(crimes['CMPLNT_FR_TM'], axis = 0, inplace = True)


# <br>***JURISDICTION_CODE*** - matched *JURISDICTION_CODE* values to *JURIS_DESC* and filled *JURISDICTION_CODE* NaNs

# In[ ]:


# Find jurisdiction codes for all jurisdictions responsible for incident
juris_desc = crimes[['JURISDICTION_CODE', 'JURIS_DESC']].drop_duplicates(subset = 'JURIS_DESC', keep = 'first').reset_index(drop = True)
juris_desc.rename(columns = {'JURIS_DESC' : 'JUR_NAME'}, inplace = True) #change column name to simplify distinction of colmumns` names
print(juris_desc)


# In[ ]:


# Fill NaN values with some number (different from any JURISDICTION_CODE).
crimes['JURISDICTION_CODE'].fillna(5, inplace = True)

# Replacement of filled NaNs values with appropriate numbers.
# I don`t understand how below line exactly works, but I found similar solution on the web (and can`t find source again to analyze).
crimes['JURISDICTION_CODE'] = crimes['JURISDICTION_CODE']                               .mask(crimes['JURISDICTION_CODE'] == 5, crimes['JURIS_DESC']                               .map(juris_desc.set_index('JUR_NAME')['JURISDICTION_CODE']))


# <br>***OFNS_DESC*** -  'OFNS_DESC' column has only 4 NaN values. Rows with those values have values in PD_DESC, so the best solution is to copy them.
# <br>***PD_DESC*** - NaN values filled with *OFNS_DESC* values

# In[ ]:


# All NaNs from 'PD_DESC' series are filled with copy of 'OFNS_DESC' values
crimes['OFNS_DESC'] = np.where(crimes['OFNS_DESC'].isnull(), crimes['PD_DESC'], crimes['OFNS_DESC']) # There is pandas equivalent of np.where -> https://stackoverflow.com/questions/38579532/pandas-equivalent-of-np-where
# And vice versa
crimes['PD_DESC'] = np.where(crimes['PD_DESC'].isnull(), crimes['OFNS_DESC'], crimes['PD_DESC'])

# Sanity check
print(crimes['OFNS_DESC'].notnull().value_counts()) 
print(crimes['PD_DESC'].notnull().value_counts())


# <br>***PRAKS_NM*** - NANs filled with 'NON APPLICABLE' statement
# <br>***STATION_NAME*** - NANs filled with 'NON APPLICABLE' statement
# <br>***TRANSIT_DISTRICT***  - NaNs filled with 0, all present values are numbers

# In[ ]:


# Fill NaNs in 'PARKS_NM', 'STATION_NAME' series
crimes['PARKS_NM'].fillna('NON APPLICABLE', inplace = True)
crimes['STATION_NAME'].fillna('NON APPLICABLE', inplace = True)

#Fill NaNs in 'TRANSIT_DISTRICT' with zeros, all different values are numbers, so it is good solution
crimes['TRANSIT_DISTRICT'].fillna(0, inplace = True)


# <br>***Latitude*** & ***Longitude*** - there are 2 rows with NaNs in these columns, so I chceked *ADDR_PCT_CD* value and found that is Bronx, picked random location in Bronx (manually) for these 2 NaNs
# <br> ***X_COORD_CD*** & ***Y_COORD_CD*** - NaNs filled with converted *Latitude* and *Longitude*
# <br> ***Lat_Lon*** - filled with latitude and longitude as a string type

# In[ ]:


# There are 2 rows without values in localization series, both are from BRONX, fill it with location from bronx according to ADDR_PCT_CD value
crimes['Latitude'].fillna(40.821054, inplace = True)
crimes['Longitude'].fillna(-73.893848, inplace = True)

# Convert coordinates from latitude, longitide to UTM - 2 BRONX values
converted = utm.from_latlon(crimes.iloc[2571,28], crimes.iloc[2571,29])
converted = list(converted)
crimes['X_COORD_CD'].fillna(converted[0], inplace = True)
crimes['Y_COORD_CD'].fillna(converted[1], inplace = True)


# Fill Lat_Lon series values
lat_lon = '(' + crimes['Latitude'].astype(str) + ', ' + crimes['Longitude'].astype(str) + ')'   # It`s important to apply "(...).astype(str)" not "str(...)" below - I made this mistake
crimes['Lat_Lon'].fillna(value = lat_lon, axis = 0, inplace = True)


# <br>***PREM_TYP_DESC*** - fill NaNs with randomly picked values from locations with appropriate content

# In[ ]:


# Creat copy to calculate proportions.
prem_typ_desc_copy = crimes['PREM_TYP_DESC'].copy(deep = True)
prem_typ_desc_copy_rand = prem_typ_desc_copy.value_counts(normalize = True).sort_values(ascending = False)

# Fill PREM_TYP_DESC values NaN values with values from locations of other incidents.
crimes['PREM_TYP_DESC'] = crimes['PREM_TYP_DESC'].apply(lambda x: np.random.choice([x for x in crimes.prem_typ_desc],
                          replace = True, p = prem_typ_desc_copy_rand ) if (x == np.nan) else x).astype(str)


# <br> Excluding ***BORO_NM*** three series with NaN values left: ***SUSP_AGE_GROUP***, ***SUSP_RACE***, ***SUSP_SEX***. Easier way to fill them is to find distributions of race, age_group and gender proportions, but for now I leave it with NaNs and add more 3 columns with filled NaNs. I plan to do it in two ways. First, like previous column I will apply random function. Second way is to try implement machine learnig to estimate age_group, race and gender of suspector. Finally I'll make a comparison of values obtained in two ways. Directly below is version with ranadomly distributed age_group,  sex and race. ***PATROL_BORO*** column also has NaNs. I'll try implement clustring algorith to fill values in next kernel. 

# In[ ]:


# Check values. 
print(crimes['SUSP_AGE_GROUP'].value_counts())


# In[ ]:


# Ascribe new age group for all suspectors except those which age group is 25-44, 18-24, 45-64, <18, or 65+ and do it in a new column
# Create column for a new variable: suspector_age_rand and fill it wit 0s
crimes['suspector_age_rand'] = pd.Series(len(crimes['SUSP_AGE_GROUP']), index = crimes.index)
crimes['suspector_age_rand'] = 0   # Fill with 0.

# Randomly stick age gruop to every user with NaN value
crimes.loc[(crimes['SUSP_AGE_GROUP'] != '25-44') | 
           (crimes['SUSP_AGE_GROUP'] != '18-24') |
           (crimes['SUSP_AGE_GROUP'] != '45-64') |
           (crimes['SUSP_AGE_GROUP'] != '65+') |
           (crimes['SUSP_AGE_GROUP'] != '<18') |
           (crimes['SUSP_AGE_GROUP'].isnull()), 'suspector_age_rand'] = np.nan
crimes.loc[(crimes['SUSP_AGE_GROUP'] == '25-44') | 
           (crimes['SUSP_AGE_GROUP'] == '18-24') |
           (crimes['SUSP_AGE_GROUP'] == '45-64') |
           (crimes['SUSP_AGE_GROUP'] == '65+') |
           (crimes['SUSP_AGE_GROUP'] == '<18'), 'suspector_age_rand'] = crimes['SUSP_AGE_GROUP']


# Create copy to calculate proportions
suspector_age_rand_copy = crimes['suspector_age_rand'].copy(deep = True)

# Fill NaN values. It wouldn`t work with NaN values, so I replaced it
crimes['suspector_age_rand'].fillna(value = 1, inplace = True)

# Obtain values for every age group
suspector_age_rand_copy.dropna(axis = 0, inplace = True)
sorted_suspector_age_rand = suspector_age_rand_copy.value_counts(normalize = True).sort_index()

# Fill NaNs (rightly ones) values in suspector_age_rand with randomly picked from random choice
crimes['suspector_age_rand'] = crimes['suspector_age_rand'].apply(lambda x: np.random.choice([x for x in sorted_suspector_age_rand.index],
                               replace = True, p = sorted_suspector_age_rand) if (x == 1) else x).astype(str)
print("Suspector age with filled NaNs normalized:\n", crimes['suspector_age_rand'].value_counts(normalize = True))


# In[ ]:


# Similar operations for suspector race.
# I wonder about writing a function here, I type similar code 2nd time...
print("Original data:\n",crimes['SUSP_RACE'].value_counts())

# Create column for new variable suspector_age.
crimes['suspector_race_rand'] = pd.Series(len(crimes['SUSP_RACE']), index=crimes.index)
crimes['suspector_race_rand'] = 0

# Randomly stick age gruop to every user with NaN value 
crimes.loc[(crimes['SUSP_RACE'] != 'BLACK') | 
           (crimes['SUSP_RACE'] != 'WHITE HISPANIC') |
           (crimes['SUSP_RACE'] != 'WHITE') |
           (crimes['SUSP_RACE'] != 'BLACK HISPANIC') |
           (crimes['SUSP_RACE'] != 'ASIAN/PAC.ISL') |
           (crimes['SUSP_RACE'] != 'AMER IND') |
           (crimes['SUSP_RACE'].isnull() == True), 'suspector_race_rand'] = np.nan
crimes.loc[(crimes['SUSP_RACE'] == 'BLACK') | 
           (crimes['SUSP_RACE'] == 'WHITE HISPANIC') |
           (crimes['SUSP_RACE'] == 'WHITE') |
           (crimes['SUSP_RACE'] == 'BLACK HISPANIC') |
           (crimes['SUSP_RACE'] == 'ASIAN/PAC.ISL') |
           (crimes['SUSP_RACE'] == 'AMER IND'), 'suspector_race_rand'] = crimes['SUSP_RACE']


# Create copy to calculate proportions.
suspector_race_rand_copy = crimes['suspector_race_rand'].copy(deep = True)

# Fill NaN values.
crimes['suspector_race_rand'].fillna(value = 1, inplace = True)

# Obtain values for every race.
suspector_race_rand_copy.dropna(axis = 0, inplace = True)
sorted_suspector_race_rand = suspector_race_rand_copy.value_counts(normalize = True).sort_index()

# Fill one values in suspector_race with randomly picked from random choice.
crimes['suspector_race_rand'] = crimes['suspector_race_rand'].apply(lambda x: np.random.choice([x for x in sorted_suspector_race_rand.index],
                                replace = True, p = sorted_suspector_race_rand) if (x == 1) else x).astype(str)
print("\nFilled NaNs normalized:\n", crimes['suspector_race_rand'].value_counts(normalize = True))


# In[ ]:


# Similar operations for suspector sex.
# I type similar code 3rd type...
print("Original data:\n", crimes['SUSP_SEX'].value_counts(dropna = False))

# Create column for new variable suspector_age.
crimes['suspector_sex_rand'] = pd.Series(len(crimes['SUSP_SEX']), index = crimes.index)
crimes['suspector_sex_rand'] = 0

# Randomly stick sex to every user with NaN value.
crimes.loc[(crimes['SUSP_SEX'] != 'M') | 
           (crimes['SUSP_SEX'] != 'F') |
           (crimes['SUSP_SEX'].isnull() == True), 'suspector_sex_rand'] = np.nan
crimes.loc[(crimes['SUSP_SEX'] == 'M') | 
           (crimes['SUSP_SEX'] == 'F'), 'suspector_sex_rand'] = crimes['SUSP_SEX']


# Create a copy to calculate proportions.
suspector_sex_rand_copy = crimes['suspector_sex_rand'].copy(deep = True)

# Fill NaN values.
crimes['suspector_sex_rand'].fillna(value = 1, inplace = True)

# Obtain values for every sex.
suspector_sex_rand_copy.dropna(axis = 0, inplace = True)
sorted_suspector_sex_rand = suspector_sex_rand_copy.value_counts(normalize = True).sort_index()

# Fill one values in suspector_sex_rand with randomly picked from random choice.
crimes['suspector_sex_rand'] = crimes['suspector_sex_rand'].apply(lambda x: np.random.choice([x for x in sorted_suspector_sex_rand.index],
                                replace = True, p = sorted_suspector_sex_rand) if (x == 1) else x).astype(str)
print("Gender proportions after filled NaNs: \n", crimes['suspector_sex_rand'].value_counts(normalize = True))


# <br> Columns ***VIC_SEX***, ***VIC_AGE_GROUP*** and ***VIC_RACE*** have inappropriate values, but I leave them now and will fill using ML in near future. It could be easily filled with random function as above. You can try.

# In[ ]:


# Informations about values of victims.
print("Age values: ", crimes['VIC_AGE_GROUP'].unique())
print("Sex values: ", crimes['VIC_SEX'].unique())
print(" Race values: ", crimes['VIC_RACE'].unique())


# <a id="6"></a> <br>
# ### WHICH BOROUGH?

# It's time to fill NaN values in BORO_NM column. I think there is one easy, but work demanding way to fill them. NY Boroughs boundaries data will be needed. It could be obtained from NYC Open Data (source: [CLICK](http://data.cityofnewyork.us/City-Government/Borough-Boundaries/tqmj-j8zm)) or directly from geopandas package which seems easier solution. In this kernel data come from NYC Open Data. 

# In[ ]:


# Values of BORO_NM colum.
crimes['BORO_NM'].value_counts(dropna = False)


# In[ ]:


# Find rows with NaN values in crimes data frame.
nan_boros = crimes[crimes['BORO_NM'].isnull()]
# nan_boros.info()


# The idea of finding BORO_NM value is simple. Longitude and latitude values of incidents from crimes data frame indicates if point of interest lies within boundries of one from 5 different boroughs. to check it it will be used *shapely* package.

# In[ ]:


# Import NYC Borough boundries shape file. It is important to read in file using geopandas instead of pandas.
boros = gpd.read_file('../input/nycboros/geo_export_366953f5-d29c-43ab-9e8e-781c703ad083.shp')
boros['boro_name'] = boros['boro_name'].str.upper()
boros


# In[ ]:


# Function to make points - credits for https://twitter.com/dangerscarf
def make_point(row):
    return Point(row.Longitude, row.Latitude)

# Go through every row, and make a point out of its lon and lat
points = crimes.apply(make_point, axis=1)

# New GeoDataFrame with old data and new geometry
crimes_geo = gpd.GeoDataFrame(crimes, geometry = points)

# It doesn't come with a CRS because it's a CSV, so let's change it. I didn`t dive deep into it,
# but I have read, that it`s needed to get right results.
crimes_geo.crs = {'init': 'epsg:4326'}

# crimes_geo.head()


# In[ ]:


# Fill NaN values in BORO_NM
def fill_boro_nan(gdf1, gdf2):
    """ 
    Function fills NaN values in BORO_NM column in geodataframe and returns gdf with filled values.
    gdf1 - geodataframe with NaN values and POINTS
    gdf2 - geodataframe with POLYGONS and borougs names
    """
    boro_list = [] # List for keeping values of BORO_NM
    for point in gdf1['geometry'][gdf1['BORO_NM'].isnull()]:   # Iterate through rows with NaN values in BORO_NM column
        for i in range(0, len(gdf2['geometry'])):   # Iterate through rows of boros data frame
            if point.within(gdf2['geometry'][i]):   # Check if incident is within boundaries of one of boroughs
                gdf1['BORO_NM'][gdf1['BORO_NM'].isnull()][i] = gdf2['boro_name'][i]   # Ascribe borough name to incident location
                boro_list.append(gdf2['boro_name'][i])   # Make a list of boroughs
            
    boro_s = pd.Series(v for v in boro_list)    # Change list to series
    # print(boro_s)
    gdf1['BORO_NM'][gdf1['BORO_NM'].isnull()] = boro_s.values    # Fill NaN values in BORO_NM
    return boro_s    # Will be usefull to check if fills are correct


# In[ ]:


#Execute
boro_s = fill_boro_nan(crimes_geo, boros)
nan_boros['BORO_NM'] = boro_s.values


# In[ ]:


# Check values
crimes_geo['BORO_NM'].value_counts(dropna = False)


# In[ ]:


# Create geodataframe to check BORO_NM
incidents = nan_boros[['BORO_NM', 'Latitude', 'Longitude']]
incidents['Coordinates'] = list(zip(incidents['Longitude'], incidents['Latitude']))
incidents['Coordinates'] = incidents['Coordinates'].apply(Point)
geo_incidents = gpd.GeoDataFrame(incidents, geometry = 'Coordinates')
#print(geo_incidents.head())


# In[ ]:


# Check if filled values of BORO_NM corresponds with incident locations.
base = boros.plot(figsize = (16, 12), edgecolor = 'k', column = 'boro_name', cmap = 'Pastel1', alpha = 0.5)
geo_incidents.plot(ax = base, marker = 'o', column = 'BORO_NM', cmap = 'brg', markersize = 25, alpha = 0.8);
# Everyhing seems to be OK!


# In[ ]:


# Drop useless values and save data frame for future use.
crimes_df = crimes_geo.drop(['CMPLNT_NUM', 'CRM_ATPT_CPTD_CD', 'KY_CD', 'OFNS_DESC', 'PREM_TYP_DESC', 'geometry', 'Lat_Lon', 'X_COORD_CD', 'Y_COORD_CD', 'RPT_DT'], axis = 1)
crimes_df.to_csv('crimes_df.csv', index = False)


# In[ ]:


# Check main data frame once again
crimes_df.info()


# <a id="7"></a> <br>
# ### FINAL TOUGHTS

# That's all in this notebook. In a few days I'm going to do the following:
# 1. Write function to fill NaNs in suspectors' columns instead of typing similar code many times like above. **Update 2018-20-09**. *I wrote that function and presented it in next kernel.*
# 2. Fill NaNs in column BORO_PATROL with some clustering algorithm.**Update 2018-24-08** *I'll filled it in second kernel - [here](https://www.kaggle.com/mihalw28/nyc-crimes-2018-random-forest-regressor-nans).*
# 3. Make static and interactive visualisations.
# 
# Thank you for your time.

# In[ ]:




