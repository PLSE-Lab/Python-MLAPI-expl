#!/usr/bin/env python
# coding: utf-8

# # Problem Statement
# 
# How do you measure justice? And how do you solve the problem of racism in policing? We look for factors that drive racial disparities in policing by analyzing census and police department deployment data. The ultimate goal is to inform police agencies where they can make improvements by identifying deployment areas where racial disparities exist and are not explainable by crime rates and poverty levels.
# 
# Our biggest challenge is automating the combination of police data, census-level data, and other socioeconomic factors. Shapefiles are unusual and messy -- which makes it difficult to, for instance, generate maps of police behavior with precinct boundary layers mixed with census layers. Police incident data are also very difficult to normalize and standardize across departments since there are no federal standards for data collection..

# # Key takeaways, what I am going to look for:
# 
# > Since we are looking for racial disparities that are not explainable by crime rates and poverty levels, we first need to look for crime rates and poverty levels, does our data provide this information?
# 
# > The goal is to see, if there are racial disparities, so we need to also look, how racial information is provided in our data sets.
# 

# # What we need to do with the data?
# 
# - match census data to police data - use census tract code as a key
# - use police data address or lat/long to get the census tract code
# - count how many incidents per census tract code, we need large enough sample, to be able to draw conclusions
# - if combining police and census data successful and if the resulting dataset statistically significant, create meaningful visual representations of the data.
# 
# # What to look out for with provided data?
# 
# - Make sure Census data and police data from the same geographic area before starting analyses
# - Not clear, if the period police data is available, is all or only a sample
# - Not clear if all geographic area included in the police data or again, just a sample
# 
# # Potential issues:
# 
# - Crime rates are not explicitly provided. Since it is not clear, if the provided police data is complete, it is not possible to derive crime rates from given data either.
# - Is crime reported local, i.e. incidents reported happening in the same place people involved are living? If not, could not really compare racial make of the population to racial distribution of people involved in police incidences, at least not at the census tract level. 
# - What is the relationship between poverty and crime? This question seems larger than the current project. 
# 

# In[ ]:


import os, sys
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from geopandas.tools import sjoin
from matplotlib import pyplot as plt


# In[ ]:


import warnings
warnings.simplefilter(action='ignore')


# In[ ]:


print(os.listdir("../input"))


# In[ ]:


# Lets look at the actual data now

data_folder = "../input/data-science-for-good/cpe-data/"

#data_folder = "C:/Users/Lilianne/Downloads/cpe-data/"
dirs = os.listdir( data_folder )

deparment_list = []

for filename in os.listdir(data_folder):
    if os.path.isdir(os.path.join(data_folder,filename)):
        deparment_list.append(filename)

print(deparment_list)    


# We see, that we have 6 departments data. Let's see, what type of data each of those folders contain.

# In[ ]:


department = "Dept_11-00091"

dirs = os.listdir( data_folder + department + "/")

for file in dirs:
    print(file)


# In[ ]:


department = "Dept_23-00089"

dirs = os.listdir( data_folder + department + "/")

for file in dirs:
    print(file)


# In[ ]:


department = "Dept_35-00103"

dirs = os.listdir( data_folder + department + "/")

for file in dirs:
    print(file)


# In[ ]:


department = "Dept_37-00027"

dirs = os.listdir( data_folder + department + "/")

for file in dirs:
    print(file)


# In[ ]:


department = "Dept_37-00049"

dirs = os.listdir( data_folder + department + "/")

for file in dirs:
    print(file)


# In[ ]:


department = "Dept_49-00009"

dirs = os.listdir( data_folder + department + "/")

for file in dirs:
    print(file)


# Only 3 out 6 departments include the police incident reports. Since I am interested in actual crime and not just shape files, I will take a closer look at those 3.

# In[ ]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[ ]:


department = "37-00049"

dirs = os.listdir( data_folder + "Dept_" + department + "/" + department + "_ACS_data/")

for file in dirs:
    print(file)


# Since I am interested in race, I will open the ACS_race-sex-age and take a look

# In[ ]:


census_race_df = pd.read_csv(data_folder + "Dept_" + department + "/" + department + "_ACS_data/" + department + "_ACS_race-sex-age/ACS_15_5YR_DP05_with_ann.csv").iloc[0:].reset_index(drop=True)

police_incident_df = pd.read_csv(data_folder + "Dept_" + department + "/" + department + "_UOF-P_2016_prepped.csv").iloc[1:].reset_index(drop=True)


# In[ ]:


census_race_df.head()


# In[ ]:


police_incident_df.head()


# We checked that Department 37-00049 data is for Dallas, TX. We will do the same for the other two too.

# In[ ]:


department = "37-00027"

dirs = os.listdir( data_folder + "Dept_" + department + "/" + department + "_ACS_data/")

census_race_df_3700027 = pd.read_csv(data_folder + "Dept_" + department + "/" + department + "_ACS_data/" + department + "_ACS_race-sex-age/ACS_15_5YR_DP05_with_ann.csv").iloc[0:].reset_index(drop=True)

police_incident_df_3700027 = pd.read_csv(data_folder + "Dept_" + department + "/" + department + "_UOF-P_2014-2016_prepped.csv").iloc[1:].reset_index(drop=True)


# In[ ]:


census_race_df_3700027.head()


# In[ ]:


police_incident_df_3700027.head()


# Looks like Department 37-00027 data has issues. Census data is from Hills County TX, whereas police data is from Austin TX. We will stop working with this data for now. 

# In[ ]:


department = "35-00103"

dirs = os.listdir( data_folder + "Dept_" + department + "/" + department + "_ACS_data/")

census_race_df_3500103 = pd.read_csv(data_folder + "Dept_" + department + "/" + department + "_ACS_data/" + department + "_ACS_race-sex-age/ACS_15_5YR_DP05_with_ann.csv").iloc[0:].reset_index(drop=True)

police_incident_df_3500103 = pd.read_csv(data_folder + "Dept_" + department + "/" + department + "_UOF-OIS-P_prepped.csv").iloc[1:].reset_index(drop=True)


# In[ ]:


census_race_df_3500103.head()


# In[ ]:


police_incident_df_3500103.head()


# Looks like department 35-00103 data for both census and police data, is from Charlotte, North Carolina. 
# 
# We now have two datasets which seem to have matching geographical data. We will proceed with those sets, trying to match census tracts to police records. 

# In[ ]:


census_shp_gdf_3500103 = gpd.read_file("../input/censusshapes3500103/tl_2010_37119_tract10.shp")


# In[ ]:


census_shp_gdf_3500103.head()


# In[ ]:


latlon_exists_index = police_incident_df_3500103[['LOCATION_LATITUDE','LOCATION_LONGITUDE']].dropna().index

police_incident_df_3500103 = police_incident_df_3500103.iloc[latlon_exists_index].reset_index(drop=True)

police_incident_df_3500103['LOCATION_LATITUDE'] = (police_incident_df_3500103['LOCATION_LATITUDE']
                                         .astype('float'))
police_incident_df_3500103['LOCATION_LONGITUDE'] = (police_incident_df_3500103['LOCATION_LONGITUDE']
                                         .astype('float'))

# important to check if order in Shapefile is Point(Longitude,Latitude)
police_incident_df_3500103['geometry'] = (police_incident_df_3500103
                                .apply(lambda x: Point(x['LOCATION_LONGITUDE'],
                                                       x['LOCATION_LATITUDE']), 
                                       axis=1))
police_incident_gdf_3500103 = gpd.GeoDataFrame(police_incident_df_3500103, geometry='geometry')
police_incident_gdf_3500103.crs = {'init' :'epsg:4326'}


# In[ ]:


police_incident_gdf_3500103.head()


# In[ ]:


# number of rows and number of columns in our dataset

police_incident_gdf_3500103.shape


# In[ ]:


max_1=police_incident_df_3500103.max()
min_1=police_incident_df_3500103.min()

print(min_1['INCIDENT_DATE'], max_1['INCIDENT_DATE'])


# In[ ]:


## plot
f, ax = plt.subplots(1, figsize=(10, 12))
census_shp_gdf_3500103.plot(ax=ax)
police_incident_gdf_3500103.plot(ax=ax, marker='*', color='black', markersize=15)
plt.title("Incident Locations and Census Tracts")
plt.show()


# As we can see from the above plot, we have almost one to one relationship in police incidents to census tracts. I will overlay the police districts to census tracts and see, if this will get us little more to go by, but with 76 incident records over span of 13 years, this is not very likely.
# 
# I have two different tracks to pursue - I can either keep working with Charlotte data and try to get more police data, so that the analyses becomes meaningful, or since we have more records for Austin TX, I will work on that data-set next.
# 
# If you find any of the above useful, I would appreciate an up-vote.
# 
# Thanks

# 
# 
