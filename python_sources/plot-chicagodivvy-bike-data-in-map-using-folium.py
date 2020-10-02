#!/usr/bin/env python
# coding: utf-8

# ### About:
# 
# #### Using Pandas DataFrame
#  In this project we will map Divvy bike stations at Chicago using Folium.
# 
# -  Rename columns
# -  Convert dtype
# -  Check null values
# -  Convert laittude and longitude into point object and plot to the map.
# 
# #### Using PySpark and SparkDataframe
# I have also created this project using pyspark. Here is the [Here is the link of the Databricks notebook](https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/8729880089806595/1158333375392443/7967251143560102/latest.html). Please make sure that you install folium package before running this notebook if you are using databricks community edition.
# 

# ### Import Data

# In[ ]:


divvy_stations = pd.read_csv("../input/Divvy_Bicycle_Stations.csv")
divvy_stations.head()


# ### Rename Columns
# Remove space between the column lables

# In[ ]:


divvy_station_columns = divvy_stations.columns
col_with_space_removed = [''.join(j for j in i.title() if not j.isspace()) for i in divvy_station_columns]
camel_cols = [col[0].lower()+col[1:] for col in col_with_space_removed ]
divvy_stations.columns =camel_cols

#display columns
divvy_stations.columns


# ### Data Preparation
# Convert datatype
# 
# -id:integer
# -stationName:string
# -docksInService:integer
# -latitude:float
# -longitude:float
# 

# In[ ]:


#sdf_typed = spark.sql("SELECT cast(id as int), stationName, cast(docksInService as int), 
                      #cast(latitude as float), cast(longitude as float)   FROM renamed_divvyStation_df")
                      
sliced_pdf = divvy_stations[["id","stationname","docksinservice","latitude","longitude" ]]                      
sliced_pdf.astype({'id': 'int32',
                  'stationname':'str',
                  'docksinservice':'int32',
                  'latitude':'float',
                  'longitude':'float'})
sliced_pdf.head()
    
                      
                      
                      


# In[ ]:


sliced_pdf.describe()


# It seems that there is no null values. 

# ### Plot data using Folium
# 

# In[ ]:


import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
from folium import IFrame

locations = sliced_pdf[['latitude', 'longitude']]
locationlist = locations.values.tolist()
len(locationlist)
locationlist[7:9]


# In[ ]:



folium_map = folium.Map([41.894722,-87.634362],  zoom_start = 13)

for point in range(0, len(locationlist)):
    folium.Marker(locationlist[point], popup=divvy_stations['stationname'][point])    .add_to(folium_map)

folium_map


# In[ ]:




