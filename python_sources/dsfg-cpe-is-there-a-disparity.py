#!/usr/bin/env python
# coding: utf-8

# ## Competition Objective:
# ![CPE](https://storage.googleapis.com/kaggle-organizations/1964/thumbnail.png?r=654)
# 
# The Center for Policing Equity (CPE) is research scientists, race and equity experts, data virtuosos, and community trainers working together to build more fair and just systems. CPE look for factors that drive racial disparities in policing by analyzing census and police department deployment data. The ultimate goal is to inform police agencies where they can make improvements by identifying deployment areas where racial disparities exist and are not explainable by crime rates and poverty levels.
# 
# I would like to thank CPE for making this data available for us to analyze and get some insights. 
# 
# ## Objective of the Notebook:
# 
# We are given information about use of force data, police department data and census-level data of different socioeconomic factors. We are given both csv files and shape files. Shapefiles are unusual and messy -- which makes it difficult to, for instance, generate maps of police behavior with precinct boundary layers mixed with census layers.
# 
# So in this notebook, let us try to combine the shape files from police precinct data and from census data to make it more comparable. We will also make some visual plots along the way and make some inferences on the way.

# In[ ]:


import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# folium for maps
import folium
from folium import plugins

# geopandas for operations on shape files
import geopandas as gpd
from shapely.geometry import Polygon
from pprint import pprint 

# plotly for other visuals
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import plotly.plotly as py
from plotly import tools
import plotly.figure_factory as ff
init_notebook_mode(connected=True)
pd.options.display.max_columns = 999


# ## Dataset Description
# 
# Before even going into the analysis part, let us first look at the datasets provided and understand the same.

# In[ ]:


os.listdir("../input/data-science-for-good/cpe-data/")


# Each of these directories include data associated with specific police departments. 
# 
# For each police department there are two sub-directories for shape files and demographic data. Some departments also include police activity (use of force) CSV files. An example structure from one of the directories can be seen below.
# 
# ```
# Example:
# Dept_37-00027
#     |- Dept_37-00027_Shapefiles
#     |- Dept_37-00027_ACS_data (American Community Survey Data)
#     |- 37-00027_UOF-P_2014-2016_prepped.csv (use of force)
# ```
# 
# If you are wondering what a shape file is, then [this link](https://doc.arcgis.com/en/arcgis-online/reference/shapefiles.htm) can help you get started. Shivam has a detailed [discussion post](https://www.kaggle.com/center-for-policing-equity/data-science-for-good/discussion/67776) on various types of file formats included in shape files.  
# 
# So ideally each directory will have three files
# * One sub-directory with the police department shape files
# * One sub-directory with American Community Survey Data
# * One file containing police incident data
# 
# 
# ## Minneapolis Police
# 
# To start with, let us explore the dataset of Minneapolis police. It is present in the folder `Dept_24-00013`. We are expecting three folders / files - one for ACS, one for use of force and one for police department shape files. Let us confirm the same.
# 
# 

# In[ ]:


os.listdir("../input/data-science-for-good/cpe-data/Dept_24-00013")


# **American Community Survey Data of Minneapolis**:
# 
# First let us have a look at the type of files present inside the survey folder. 

# In[ ]:


os.listdir("../input/data-science-for-good/cpe-data/Dept_24-00013/24-00013_ACS_data/")


# As we can see, the census folder consists of the following information
#  * Education attainment
#  * Education attainment over age 25
#  * Income
#  * Employment
#  * Poverty
#  * Race - Sex - Age
#  * Owner occupied housing
# 
# Let us look at the top few lines of the Race - Sex - Age file - `24-00013_ACS_race-sex-age` to get an undersstanding.

# In[ ]:


fname = "../input/data-science-for-good/cpe-data/Dept_24-00013/24-00013_ACS_data/24-00013_ACS_race-sex-age/ACS_15_5YR_DP05_with_ann.csv"
acs_race_df = pd.read_csv(fname)
acs_race_clean_df = acs_race_df.loc[1:,:].reset_index(drop=True)

acs_race_clean_df["CT"] = acs_race_clean_df["GEO.display-label"].apply(lambda x: x.split("Tract ")[1].split(",")[0].strip())

acs_race_df.head()


# As we can see, minneapolis data comes under the Hennepin county of Minnesota state. 
# 
# ACS survey is generally done at census tract level and so here each row has information for each census tract like race, age and sex information. From Wiki, we can see that census tract is a geographic region defined for the purpose of taking a census.
# 
# The below diagram from ACS website gives a much more clear picture about a Census Tract.
# 
# ![ACS level](https://www.census.gov/content/census/en/programs-surveys/acs/geography-acs/concepts-definitions/jcr:content/par/expandablelist/section_1/image.img.576.medium.png/1443049671759.png)
# 
# ** Use of Force Data:**
# 
# From [Use of force wiki page](https://en.wikipedia.org/wiki/Use_of_force) - The use of force, in the context of law enforcement, may be defined as the "amount of effort required by police to compel compliance by an unwilling subject" 
# 
# Now let us look at the top few rows of Use of Force data of Minneapolis department

# In[ ]:


force_df = pd.read_csv("../input/data-science-for-good/cpe-data/Dept_24-00013/24-00013_UOF_2008-2017_prepped.csv")
force_clean_df = force_df.loc[1:,:].reset_index(drop=True)
force_df.head()


# Use of force dataset has details about latitude, longitude of the incident along with date, time and type of force used.
# 
# 
# **Minneapolis Police Precincts / Districts Shape files:**
# 
# Now let us look the top few rows of the shape file of Minneapolis Police Precincts.

# In[ ]:


fname = "../input/data-science-for-good/cpe-data/Dept_24-00013/24-00013_Shapefiles/Minneapolis_Police_Precincts.shp"
police_df = gpd.read_file(fname)
police_df.head()


# Now let us plot them in the map and see the same.

# In[ ]:


mapa = folium.Map([44.99, -93.27], height=500, zoom_start=11, tiles='Stamen Terrain')
folium.GeoJson(police_df).add_to(mapa)
mapa


# As we can see there are 5 police precincts in Minneapolis.
# 
# Now let us also plot the co-ordinates from "Use of force" data on top of police districts. Since the number of rows are huge for folium plot, let us plot only the recent 2000 incidents on the map. 

# In[ ]:


mapa = folium.Map([44.99, -93.27], height=500, zoom_start=11, tiles='Stamen Terrain')

folium.GeoJson(police_df).add_to(mapa)

locations_df = force_clean_df[["LOCATION_LATITUDE", "LOCATION_LONGITUDE"]].copy()
notna = locations_df[['LOCATION_LATITUDE','LOCATION_LONGITUDE']].dropna().index
locations_df = locations_df.iloc[notna].reset_index(drop=True)
locations_df["LOCATION_LATITUDE"] = locations_df["LOCATION_LATITUDE"].astype('float')
locations_df["LOCATION_LONGITUDE"] = locations_df["LOCATION_LONGITUDE"].astype('float')
locationlist = locations_df.values.tolist()[-2000:]
for point in range(0, len(locationlist)):
    #folium.Marker(locationlist[point], popup=df_counters['Name'][point], icon=folium.Icon(color='darkblue', icon_color='white', icon='male', angle=0, prefix='fa')).add_to(marker_cluster)
    folium.CircleMarker(locationlist[point], radius=0.1, color='red').add_to(mapa)

mapa 


# We can also plot the geopandas dataframe as such without using folium maps. It will give the polygons alone.

# In[ ]:


f, ax = plt.subplots(1, figsize=(10, 8))
police_df.plot(column="PRECINCT", ax=ax, cmap='Accent',legend=True);
plt.title("Districts : Minneapolis Police Precincts")
plt.show()


# ** ACS shape files:**
# 
# Now to combine the Police precincts data with the survey data, we need the geographical information of each of the census tracts as they can be different from the police districts.
# 
# Thankfully, we can download the shape files for each of the census tracts from the ACS website using this [link](https://www.census.gov/geo/maps-data/data/cbf/cbf_tracts.html).
# 
# I have uploaded some of them into a Kaggle dataset which can be accessed [here](https://www.kaggle.com/sudalairajkumar/dsfg-cpe-acs-shape-files)
# 
# Let us read the shape file of Minnesota state and filter the Hennepin county from the same. The top few rows are:  

# In[ ]:


fname = "../input/dsfg-cpe-acs-shape-files/cb_2017_27_tract_500k/cb_2017_27_tract_500k.shp"
acs_df = gpd.read_file(fname)
acs_df = acs_df[acs_df["COUNTYFP"]=="053"].reset_index()
acs_df.head()


# `NAME` column is the census tract id and this can be used to combine this with the ACS survey data we have.
# 
# Now let us plot them on a map and see.

# In[ ]:


mapa = folium.Map([45.04, -93.47], height=600, zoom_start=10, tiles='Stamen Terrain')
folium.GeoJson(acs_df).add_to(mapa)
mapa


# As we can see, the shape file from the police department and the shape file from the ACS survey does not cover the exact same locations.  Infact, the survey shape file location seem to be a super set of the police district data. So we need only those census tracts which are present in each of the police precincts.
# 
# ### Combining Police & Census Shape files:
# 
# To take an example, say police district 1 has 100% of Census Tract 1 (CT1) and 50% of CT2 and 25% of CT3. And if there are 100 people of Race A in each of these CTs, then we need to do the following computation to get the overall number of people of Race A in the police district 
# ```
# = 1.*100 (for CT1) + 0.5*100 (for CT2) + 0.25*100 (for CT3)
# = 100 + 50 + 25
# = 175
# ```
# 
# ### Minneapolis Police District 1:
# For simplicity, let us just conecntrate on Police district 1 of Minneapolis data and get this done. 

# In[ ]:


mapa = folium.Map([44.98, -93.27], height=500, zoom_start=13, tiles='Stamen Terrain')
folium.GeoJson(police_df.loc[0:0,:], style_function= lambda x:{'color':'red'}).add_to(mapa)
mapa


# Now let us use the ACS shape files and get the census tracts and the percentage of area in each census tracts that come under the police district 1.

# In[ ]:


police_gdf = gpd.GeoDataFrame(police_df["geometry"])
acs_gdf = gpd.GeoDataFrame(acs_df["geometry"])

print("Following Census Tracts are present in the Minneapolis Police District 1 : ")
acs_police_df = []
for i in range(acs_gdf.shape[0]):
    a = (police_gdf['geometry'][0]).intersection(acs_gdf['geometry'][i])
    if a.area != 0:
        print("CT :", acs_df['NAME'][i], " and the percentage of area is :", (a.area / acs_gdf['geometry'][i].area)*100)
        acs_police_df.append([acs_df['NAME'][i], (a.area / acs_gdf['geometry'][i].area)])
acs_police_df = pd.DataFrame(acs_police_df)
acs_police_df.columns = ["NAME", "PercentageArea"]


# Let us now overlay the police district 1 with the census tracts we got from last step and plot it in the map.

# In[ ]:


mapa = folium.Map([44.98, -93.27], height=500, zoom_start=13, tiles='Stamen Terrain')
folium.GeoJson(police_df.loc[0:0,:], style_function= lambda x:{'color':'red'}).add_to(mapa)
folium.GeoJson(acs_df[acs_df["NAME"].isin(acs_police_df["NAME"].values)], style_function= lambda x:{'color':'green'}).add_to(mapa)
mapa


# 
# This is cool.! Now we can get the census information from these places and based on the percentage area covered, we can multiply the census values with percentage covered to get the numbers. (Not an accurate method, but some approximation is better than nothing)
# 
# Now let us get the number of people based on race in this police district and also the 'use of force' distribution by race in this police dsitrict and plot the same. 

# In[ ]:


# pie chart based on population
acs_police_race_df = acs_race_clean_df[acs_race_clean_df["CT"].isin(acs_police_df["NAME"].values)].reset_index(drop=True)
acs_police_race_df = pd.merge(acs_police_race_df, acs_police_df, left_on=["CT"], right_on=["NAME"])

cols_to_use = ["HC01_VC49", "HC01_VC50", "HC01_VC51", "HC01_VC56", "HC01_VC69"]
col_names = ["White", "Black", "Native American", "Asian", "Other / Mixed Race"]
color_names = ["LightGrey", "Black", "Red", "Orange", "Green"]

race_count = []
for i, col in enumerate(cols_to_use):
    race_count.append([col_names[i], np.round((acs_police_race_df[col].astype(float) * acs_police_race_df["PercentageArea"]).sum())])
race_count_df = pd.DataFrame(race_count)
race_count_df.columns = ["race", "count"]

labels = (np.array(race_count_df["race"].values))
temp_series = race_count_df["count"]
sizes = (np.array((temp_series / temp_series.sum())*100))

trace0 = go.Pie(labels=labels, 
                values=sizes,
                domain=dict(x=[0,0.48]),
                marker=dict(
                   colors=color_names
               ),
              )

## Pie chart based on use of force
temp_series = force_clean_df["SUBJECT_RACE"][force_clean_df["LOCATION_DISTRICT"]=="1"].value_counts().head(5)
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))
color_map = {"Black":"Black", "White":"LightGrey", "Other / Mixed Race":"Green", "Asian":"Orange", "Native American":"Red"}
color_names = []
for i in labels:
    color_names.append(color_map[i])

trace1 = go.Pie(labels=labels, 
               values=sizes,
                domain=dict(x=[0.52,1]),
               marker=dict(
                   colors=color_names
               ),
              )
ann1 = dict(font=dict(size=12),
            showarrow=False,
            text='Census Population Distribution by Race ',
            # Specify text position (place text in a hole of pie)
            x=0.1,
            y=1.1,
            )
ann2 = dict(font=dict(size=12),
            showarrow=False,
            text='Use of Force Distribution by Race',
            # Specify text position (place text in a hole of pie)
            x=0.9,
            y=1.1,
            )
layout = go.Layout(title ='Minneapolis Police District 1',
                   annotations=[ann1,ann2],
                   # Hide legend if you want
                   #showlegend=False
                   )

data = [trace0, trace1]
fig = go.Figure(data=data,layout=layout)
# Plot the plot and save the file in your Python script directory
iplot(fig, filename='subplot_pie_chart.html')


# ### Minneapolis Police District 2
# 
# ####  Census & Police Shape files intersection plot

# In[ ]:


police_gdf = gpd.GeoDataFrame(police_df["geometry"])
acs_gdf = gpd.GeoDataFrame(acs_df["geometry"])

POLICE_DIST_ROW = 1

#print("Following Census Tracts are present in the Minneapolis Police District 1 : ")
acs_police_df = []
for i in range(acs_gdf.shape[0]):
    a = (police_gdf['geometry'][POLICE_DIST_ROW]).intersection(acs_gdf['geometry'][i])
    if a.area != 0:
        #print("CT :", acs_df['NAME'][i], " and the percentage of area is :", (a.area / acs_gdf['geometry'][i].area)*100)
        acs_police_df.append([acs_df['NAME'][i], (a.area / acs_gdf['geometry'][i].area)])
acs_police_df = pd.DataFrame(acs_police_df)
acs_police_df.columns = ["NAME", "PercentageArea"]

mapa = folium.Map([44.98, -93.27], height=500, zoom_start=12, tiles='Stamen Terrain')
folium.GeoJson(police_df.loc[POLICE_DIST_ROW:POLICE_DIST_ROW,:], style_function= lambda x:{'color':'red'}).add_to(mapa)
folium.GeoJson(acs_df[acs_df["NAME"].isin(acs_police_df["NAME"].values)], style_function= lambda x:{'color':'green'}).add_to(mapa)
mapa


# In[ ]:


# pie chart based on population
acs_police_race_df = acs_race_clean_df[acs_race_clean_df["CT"].isin(acs_police_df["NAME"].values)].reset_index(drop=True)
acs_police_race_df = pd.merge(acs_police_race_df, acs_police_df, left_on=["CT"], right_on=["NAME"])

cols_to_use = ["HC01_VC49", "HC01_VC50", "HC01_VC51", "HC01_VC56", "HC01_VC69"]
col_names = ["White", "Black", "Native American", "Asian", "Other / Mixed Race"]
color_names = ["LightGrey", "Black", "Red", "Orange", "Green"]

race_count = []
for i, col in enumerate(cols_to_use):
    race_count.append([col_names[i], np.round((acs_police_race_df[col].astype(float) * acs_police_race_df["PercentageArea"]).sum())])
race_count_df = pd.DataFrame(race_count)
race_count_df.columns = ["race", "count"]

labels = (np.array(race_count_df["race"].values))
temp_series = race_count_df["count"]
sizes = (np.array((temp_series / temp_series.sum())*100))

trace0 = go.Pie(labels=labels, 
                values=sizes,
                domain=dict(x=[0,0.48]),
                marker=dict(
                   colors=color_names
               ),
              )

## Pie chart based on use of force
POLICE_DISTRICT = "2"
temp_series = force_clean_df["SUBJECT_RACE"][force_clean_df["LOCATION_DISTRICT"]==POLICE_DISTRICT].value_counts().head(5)
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))
color_map = {"Black":"Black", "White":"LightGrey", "Other / Mixed Race":"Green", "Asian":"Orange", "Native American":"Red"}
color_names = []
for i in labels:
    color_names.append(color_map[i])
    
trace1 = go.Pie(labels=labels, 
               values=sizes,
                domain=dict(x=[0.52,1]),
               marker=dict(
                   colors=color_names
               ),
              )
ann1 = dict(font=dict(size=12),
            showarrow=False,
            text='Census Population Distribution by Race ',
            # Specify text position (place text in a hole of pie)
            x=0.1,
            y=1.1,
            )
ann2 = dict(font=dict(size=12),
            showarrow=False,
            text='Use of Force Distribution by Race',
            # Specify text position (place text in a hole of pie)
            x=0.9,
            y=1.1,
            )
layout = go.Layout(title ='Minneapolis Police District '+POLICE_DISTRICT,
                   annotations=[ann1,ann2],
                   # Hide legend if you want
                   #showlegend=False
                   )

data = [trace0, trace1]
fig = go.Figure(data=data,layout=layout)
# Plot the plot and save the file in your Python script directory
iplot(fig, filename='subplot_pie_chart.html')


# ### Minneapolis District 3

# In[ ]:


police_gdf = gpd.GeoDataFrame(police_df["geometry"])
acs_gdf = gpd.GeoDataFrame(acs_df["geometry"])

POLICE_DIST_ROW = 2
#print("Following Census Tracts are present in the Minneapolis Police District 1 : ")
acs_police_df = []
for i in range(acs_gdf.shape[0]):
    a = (police_gdf['geometry'][POLICE_DIST_ROW]).intersection(acs_gdf['geometry'][i])
    if a.area != 0:
        #print("CT :", acs_df['NAME'][i], " and the percentage of area is :", (a.area / acs_gdf['geometry'][i].area)*100)
        acs_police_df.append([acs_df['NAME'][i], (a.area / acs_gdf['geometry'][i].area)])
acs_police_df = pd.DataFrame(acs_police_df)
acs_police_df.columns = ["NAME", "PercentageArea"]

# pie chart based on population
acs_police_race_df = acs_race_clean_df[acs_race_clean_df["CT"].isin(acs_police_df["NAME"].values)].reset_index(drop=True)
acs_police_race_df = pd.merge(acs_police_race_df, acs_police_df, left_on=["CT"], right_on=["NAME"])

cols_to_use = ["HC01_VC49", "HC01_VC50", "HC01_VC51", "HC01_VC56", "HC01_VC69"]
col_names = ["White", "Black", "Native American", "Asian", "Other / Mixed Race"]
color_names = ["LightGrey", "Black", "Red", "Orange", "Green"]

race_count = []
for i, col in enumerate(cols_to_use):
    race_count.append([col_names[i], np.round((acs_police_race_df[col].astype(float) * acs_police_race_df["PercentageArea"]).sum())])
race_count_df = pd.DataFrame(race_count)
race_count_df.columns = ["race", "count"]

labels = (np.array(race_count_df["race"].values))
temp_series = race_count_df["count"]
sizes = (np.array((temp_series / temp_series.sum())*100))

trace0 = go.Pie(labels=labels, 
                values=sizes,
                domain=dict(x=[0,0.48]),
                marker=dict(
                   colors=color_names
               ),
              )

## Pie chart based on use of force
POLICE_DISTRICT = "3"
temp_series = force_clean_df["SUBJECT_RACE"][force_clean_df["LOCATION_DISTRICT"]==POLICE_DISTRICT].value_counts().head(5)
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))
color_map = {"Black":"Black", "White":"LightGrey", "Other / Mixed Race":"Green", "Asian":"Orange", 
             "Native American":"Red", "not recorded":"blue"}
color_names = []
for i in labels:
    color_names.append(color_map[i])
    
trace1 = go.Pie(labels=labels, 
               values=sizes,
                domain=dict(x=[0.52,1]),
               marker=dict(
                   colors=color_names
               ),
              )
ann1 = dict(font=dict(size=12),
            showarrow=False,
            text='Census Population Distribution by Race ',
            # Specify text position (place text in a hole of pie)
            x=0.1,
            y=1.1,
            )
ann2 = dict(font=dict(size=12),
            showarrow=False,
            text='Use of Force Distribution by Race',
            # Specify text position (place text in a hole of pie)
            x=0.9,
            y=1.1,
            )
layout = go.Layout(title ='Minneapolis Police District '+POLICE_DISTRICT,
                   annotations=[ann1,ann2],
                   # Hide legend if you want
                   #showlegend=False
                   )

data = [trace0, trace1]
fig = go.Figure(data=data,layout=layout)
# Plot the plot and save the file in your Python script directory
iplot(fig, filename='subplot_pie_chart.html')


# ### Minneapolis District 4

# In[ ]:


police_gdf = gpd.GeoDataFrame(police_df["geometry"])
acs_gdf = gpd.GeoDataFrame(acs_df["geometry"])

POLICE_DIST_ROW = 3
#print("Following Census Tracts are present in the Minneapolis Police District 1 : ")
acs_police_df = []
for i in range(acs_gdf.shape[0]):
    a = (police_gdf['geometry'][POLICE_DIST_ROW]).intersection(acs_gdf['geometry'][i])
    if a.area != 0:
        #print("CT :", acs_df['NAME'][i], " and the percentage of area is :", (a.area / acs_gdf['geometry'][i].area)*100)
        acs_police_df.append([acs_df['NAME'][i], (a.area / acs_gdf['geometry'][i].area)])
acs_police_df = pd.DataFrame(acs_police_df)
acs_police_df.columns = ["NAME", "PercentageArea"]

# pie chart based on population
acs_police_race_df = acs_race_clean_df[acs_race_clean_df["CT"].isin(acs_police_df["NAME"].values)].reset_index(drop=True)
acs_police_race_df = pd.merge(acs_police_race_df, acs_police_df, left_on=["CT"], right_on=["NAME"])

cols_to_use = ["HC01_VC49", "HC01_VC50", "HC01_VC51", "HC01_VC56", "HC01_VC69"]
col_names = ["White", "Black", "Native American", "Asian", "Other / Mixed Race"]
color_names = ["LightGrey", "Black", "Red", "Orange", "Green"]

race_count = []
for i, col in enumerate(cols_to_use):
    race_count.append([col_names[i], np.round((acs_police_race_df[col].astype(float) * acs_police_race_df["PercentageArea"]).sum())])
race_count_df = pd.DataFrame(race_count)
race_count_df.columns = ["race", "count"]

labels = (np.array(race_count_df["race"].values))
temp_series = race_count_df["count"]
sizes = (np.array((temp_series / temp_series.sum())*100))

trace0 = go.Pie(labels=labels, 
                values=sizes,
                domain=dict(x=[0,0.48]),
                marker=dict(
                   colors=color_names
               ),
              )

## Pie chart based on use of force
POLICE_DISTRICT = "4"
temp_series = force_clean_df["SUBJECT_RACE"][force_clean_df["LOCATION_DISTRICT"]==POLICE_DISTRICT].value_counts().head(5)
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))
color_map = {"Black":"Black", "White":"LightGrey", "Other / Mixed Race":"Green", "Asian":"Orange", 
             "Native American":"Red", "not recorded":"blue"}
color_names = []
for i in labels:
    color_names.append(color_map[i])
    
trace1 = go.Pie(labels=labels, 
               values=sizes,
                domain=dict(x=[0.52,1]),
               marker=dict(
                   colors=color_names
               ),
              )
ann1 = dict(font=dict(size=12),
            showarrow=False,
            text='Census Population Distribution by Race ',
            # Specify text position (place text in a hole of pie)
            x=0.1,
            y=1.1,
            )
ann2 = dict(font=dict(size=12),
            showarrow=False,
            text='Use of Force Distribution by Race',
            # Specify text position (place text in a hole of pie)
            x=0.9,
            y=1.1,
            )
layout = go.Layout(title ='Minneapolis Police District '+POLICE_DISTRICT,
                   annotations=[ann1,ann2],
                   # Hide legend if you want
                   #showlegend=False
                   )

data = [trace0, trace1]
fig = go.Figure(data=data,layout=layout)
# Plot the plot and save the file in your Python script directory
iplot(fig, filename='subplot_pie_chart.html')


# ### Minneapolis District 5

# In[ ]:


police_gdf = gpd.GeoDataFrame(police_df["geometry"])
acs_gdf = gpd.GeoDataFrame(acs_df["geometry"])

POLICE_DIST_ROW = 4
#print("Following Census Tracts are present in the Minneapolis Police District 1 : ")
acs_police_df = []
for i in range(acs_gdf.shape[0]):
    a = (police_gdf['geometry'][POLICE_DIST_ROW]).intersection(acs_gdf['geometry'][i])
    if a.area != 0:
        #print("CT :", acs_df['NAME'][i], " and the percentage of area is :", (a.area / acs_gdf['geometry'][i].area)*100)
        acs_police_df.append([acs_df['NAME'][i], (a.area / acs_gdf['geometry'][i].area)])
acs_police_df = pd.DataFrame(acs_police_df)
acs_police_df.columns = ["NAME", "PercentageArea"]

# pie chart based on population
acs_police_race_df = acs_race_clean_df[acs_race_clean_df["CT"].isin(acs_police_df["NAME"].values)].reset_index(drop=True)
acs_police_race_df = pd.merge(acs_police_race_df, acs_police_df, left_on=["CT"], right_on=["NAME"])

cols_to_use = ["HC01_VC49", "HC01_VC50", "HC01_VC51", "HC01_VC56", "HC01_VC69"]
col_names = ["White", "Black", "Native American", "Asian", "Other / Mixed Race"]
color_names = ["LightGrey", "Black", "Red", "Orange", "Green"]

race_count = []
for i, col in enumerate(cols_to_use):
    race_count.append([col_names[i], np.round((acs_police_race_df[col].astype(float) * acs_police_race_df["PercentageArea"]).sum())])
race_count_df = pd.DataFrame(race_count)
race_count_df.columns = ["race", "count"]

labels = (np.array(race_count_df["race"].values))
temp_series = race_count_df["count"]
sizes = (np.array((temp_series / temp_series.sum())*100))

trace0 = go.Pie(labels=labels, 
                values=sizes,
                domain=dict(x=[0,0.48]),
                marker=dict(
                   colors=color_names
               ),
              )

## Pie chart based on use of force
POLICE_DISTRICT = "5"
temp_series = force_clean_df["SUBJECT_RACE"][force_clean_df["LOCATION_DISTRICT"]==POLICE_DISTRICT].value_counts().head(5)
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))
color_map = {"Black":"Black", "White":"LightGrey", "Other / Mixed Race":"Green", "Asian":"Orange", 
             "Native American":"Red", "not recorded":"blue"}
color_names = []
for i in labels:
    color_names.append(color_map[i])
    
trace1 = go.Pie(labels=labels, 
               values=sizes,
                domain=dict(x=[0.52,1]),
               marker=dict(
                   colors=color_names
               ),
              )
ann1 = dict(font=dict(size=12),
            showarrow=False,
            text='Census Population Distribution by Race ',
            # Specify text position (place text in a hole of pie)
            x=0.1,
            y=1.1,
            )
ann2 = dict(font=dict(size=12),
            showarrow=False,
            text='Use of Force Distribution by Race',
            # Specify text position (place text in a hole of pie)
            x=0.9,
            y=1.1,
            )
layout = go.Layout(title ='Minneapolis Police District '+POLICE_DISTRICT,
                   annotations=[ann1,ann2],
                   # Hide legend if you want
                   #showlegend=False
                   )

data = [trace0, trace1]
fig = go.Figure(data=data,layout=layout)
# Plot the plot and save the file in your Python script directory
iplot(fig, filename='subplot_pie_chart.html')


# ### St. Paul Police - Ramsey County - Minnesota:
# 
# The files for the Ramsey county is present in the folder `Dept_24-00098`

# In[ ]:


os.listdir("../input/data-science-for-good/cpe-data/Dept_24-00098/")


# #### St. Paul Police Shape file plot

# In[ ]:


### American community survey data
fname = "../input/data-science-for-good/cpe-data/Dept_24-00098/24-00098_ACS_data/24-00098_ACS_race-sex-age/ACS_15_5YR_DP05_with_ann.csv"
acs_race_df = pd.read_csv(fname)
acs_race_clean_df = acs_race_df.loc[1:,:].reset_index(drop=True)
acs_race_clean_df["CT"] = acs_race_clean_df["GEO.display-label"].apply(lambda x: x.split("Tract ")[1].split(",")[0].strip())

force_df = pd.read_csv("../input/data-science-for-good/cpe-data/Dept_24-00098/24-00098_Vehicle-Stops-data.csv")
force_clean_df = force_df.loc[1:,:].reset_index(drop=True)

fname = "../input/data-science-for-good/cpe-data/Dept_24-00098/24-00098_Shapefiles/StPaul_geo_export_6646246d-0f26-48c5-a924-f5a99bb51c47.shp"
police_df = gpd.read_file(fname)

#-93.18, 44.95
mapa = folium.Map([44.99, -93.08], height=500, zoom_start=11, tiles='Stamen Toner')
folium.GeoJson(police_df).add_to(mapa)
mapa


# #### ACS community survey shape file plot

# In[ ]:


fname = "../input/dsfg-cpe-acs-shape-files/cb_2017_27_tract_500k/cb_2017_27_tract_500k.shp"
acs_df = gpd.read_file(fname)
acs_df = acs_df[acs_df["COUNTYFP"]=="123"].reset_index()
acs_df.head()

mapa = folium.Map([44.99, -93.08], height=600, zoom_start=11, tiles='Stamen Toner')
folium.GeoJson(acs_df).add_to(mapa)
mapa


# The below are the input features we need to give to analyze a given police district data.
# 
# The column name of the police district should be variable `police_area_column`. we also need to specify the name of the police district in variable `police_area_value`
# 
# We also need to specify the name of the column having shape information in `police_shp_column`. The police district which we are interested in should be specified at `police_district`
# 

# In[ ]:


### Config ###
police_area_column = "gridnum"
police_area_value = "1"
police_shp_column = "geometry"

police_district = "1"


# In[ ]:


police_gdf = gpd.GeoDataFrame(police_df[police_shp_column])
acs_gdf = gpd.GeoDataFrame(acs_df["geometry"])

#print("Following Census Tracts are present in the Minneapolis Police District 1 : ")
acs_police_df = []
for i in range(acs_gdf.shape[0]):
    a = (police_gdf[police_shp_column][police_df[police_area_column]==police_area_value].iloc[0]).intersection(acs_gdf['geometry'][i])
    if a.area != 0:
        #print("CT :", acs_df['NAME'][i], " and the percentage of area is :", (a.area / acs_gdf['geometry'][i].area)*100)
        acs_police_df.append([acs_df['NAME'][i], (a.area / acs_gdf['geometry'][i].area)])
acs_police_df = pd.DataFrame(acs_police_df)
acs_police_df.columns = ["NAME", "PercentageArea"]

# pie chart based on population
acs_police_race_df = acs_race_clean_df[acs_race_clean_df["CT"].isin(acs_police_df["NAME"].values)].reset_index(drop=True)
acs_police_race_df = pd.merge(acs_police_race_df, acs_police_df, left_on=["CT"], right_on=["NAME"])

cols_to_use = ["HC01_VC49", "HC01_VC50", "HC01_VC51", "HC01_VC56", "HC01_VC69"]
col_names = ["White", "Black", "Native American", "Asian", "Other / Mixed Race"]
color_names = ["LightGrey", "Black", "Red", "Orange", "Green"]

race_count = []
for i, col in enumerate(cols_to_use):
    race_count.append([col_names[i], np.round((acs_police_race_df[col].astype(float) * acs_police_race_df["PercentageArea"]).sum())])
race_count_df = pd.DataFrame(race_count)
race_count_df.columns = ["race", "count"]

labels = (np.array(race_count_df["race"].values))
temp_series = race_count_df["count"]
sizes = (np.array((temp_series / temp_series.sum())*100))

trace0 = go.Pie(labels=labels, 
                values=sizes,
                domain=dict(x=[0,0.48]),
                marker=dict(
                   colors=color_names
               ),
              )

## Pie chart based on use of force
temp_series = force_clean_df["SUBJECT_RACE"][force_clean_df["LOCATION_DISTRICT"]==police_district].value_counts().head(5)
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))
color_map = {"Black":"Black", "White":"LightGrey", "Other / Mixed Race":"Green", "Asian":"Orange", 
             "Native American":"Red", "not recorded":"blue", "Latino":"green", "No Data":"yellow"}
color_names = []
for i in labels:
    color_names.append(color_map[i])
    
trace1 = go.Pie(labels=labels, 
               values=sizes,
                domain=dict(x=[0.52,1]),
               marker=dict(
                   colors=color_names
               ),
              )
ann1 = dict(font=dict(size=12),
            showarrow=False,
            text='Census Population Distribution by Race ',
            # Specify text position (place text in a hole of pie)
            x=0.1,
            y=1.1,
            )
ann2 = dict(font=dict(size=12),
            showarrow=False,
            text='Use of Force Distribution by Race',
            # Specify text position (place text in a hole of pie)
            x=0.9,
            y=1.1,
            )
layout = go.Layout(title ='St Paul Police District '+police_district,
                   annotations=[ann1,ann2],
                   # Hide legend if you want
                   #showlegend=False
                   )

data = [trace0, trace1]
fig = go.Figure(data=data,layout=layout)
# Plot the plot and save the file in your Python script directory
iplot(fig, filename='subplot_pie_chart.html')


# ### Conclusion & Next steps:
#  * This gives an example of how we can combine the police shape files with ACS shape files to figure out the racial disparity
#  * We could also similar analysis on other disparities like gender, education etc
#  * We could extend the above code base for other police districts as well.

# **References:**
# 
# A big shoutout to the following kernels / links which helped me get started with the GIS data exploration / this dataset.
# 
# 1. https://www.kaggle.com/shivamb/hunting-for-insights-geo-maps
# 2. https://georgetsilva.github.io/posts/mapping-points-with-folium/
# 3. https://www.kaggle.com/dsholes/confused-start-here
# 4. https://www.kaggle.com/crawford/another-world-famous-starter-kernel-by-chris

# In[ ]:




