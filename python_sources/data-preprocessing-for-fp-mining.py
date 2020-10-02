#!/usr/bin/env python
# coding: utf-8

# Next step [ACS Racism index and policy department information](https://www.kaggle.com/linfaner2/acs-racism-index-and-policy-department-information)

# # Overview

# The problem provided by The Center for Policing Equity(CPE) aims to find a way of measuring justice and solve the problem of racism in policing. In order to find clear and significant information to determine which factor or what kinds of areas are likely to cause or have racial disparity and to extract the essential information from these data, the first thing we need to do is to have a clear understanding of our dataset.

# # Data Set

# The dataset of Data Science for Good contains both the police records and the geometry information files among 12 cities. To make good use of our data, the first thing to do is to map our arrest record with our police department data and geometry data. When first having a glance at the arrest record dataset most of them recorded the longitude and latitude of the crime happened location, which can be mapped with the and shapefiles by checking if the polygon contains the arrested record points. By doing so, we can exclude the arrest point (also can be reagred as outliers ) which are not laied in under given police sectors. 
# 
# The ACS dataset constains the census information of each [Census Tract](https://en.wikipedia.org/wiki/Census_tract). To combine the ACS information with the arrest record, we introduce the knowledge of Racism Index which measures the Segregation, Education, Economics, Employment and provides a better and comprehensible measurement of racial disparity according to Segregation, Education, Economics, and Employment factors. The index calculation will be detailed explaned in the other kernal. By checking the geometry relationship of each census tract and our arrest record data, we can update the GEOid to each arrest record and in this way, the census data and arrest record are mapped with each other.

# # Methodology of Mining

# Mapped our arrest data with the ACS data, we can update the arrest ratio on each tract, and by mapping with the  police sectors data, we then can get the arrest ratio of each race under different police sectors. After getting the combined dataset, we decide to use frequent pattern mining to see if there exist any strong association rules and also by doing so can eliminate unrelated variables.

# In[ ]:


import numpy as np
import geopandas as gpd
from geopandas.tools import sjoin
from matplotlib import pyplot as plt
import pandas as pd
from shapely.geometry import Point
import os
import seaborn as sns
import folium
from folium import plugins
import geopandas as gpd
import plotly.graph_objs as go
import plotly.plotly as py
from shapely.geometry import Point


# Any results you write to the current directory are saved as output.


# The data folder of our main data: 

# In[ ]:


os.listdir("../input/data-science-for-good/cpe-data")


# From these 12 department, we choose to use only 5 of them:
# Dept-2400013 Minnesota
# Dept-2400098 St Paul
# Dept-3700027 Austin
# Dept-3500016 Orlando
# Dept-3700049 Dallas

# # Austin

# First, let us have a look at the arrest file. For future mapping, we need to convert the longitude and latitude to the shaply geometry point.

# Austin arrest file

# In[ ]:


force_df = pd.read_csv('../input/data-science-for-good/cpe-data//Dept_37-00027'+
                         '/37-00027_UOF-P_2014-2016_prepped.csv')
force_clean_df = force_df.loc[1:].reset_index(drop=True)
force_clean_df ['LOCATION_LONGITUDE']= pd.to_numeric(force_clean_df['LOCATION_LONGITUDE'], downcast='float') 
force_clean_df ['LOCATION_LATITUDE']= pd.to_numeric(force_clean_df['LOCATION_LATITUDE'], downcast='float') 
force_clean_df = force_clean_df[np.isfinite(force_clean_df['LOCATION_LONGITUDE'])]
force_clean_df=force_clean_df[force_clean_df['LOCATION_LONGITUDE']!=0].reset_index(drop=True)
foo=lambda x: Point(x['LOCATION_LONGITUDE'],x['LOCATION_LATITUDE'])
force_clean_df['geometry'] = (force_clean_df.apply(foo, axis=1))
force_clean_df = gpd.GeoDataFrame(force_clean_df, geometry='geometry')
force_clean_df.crs = {'init' :'epsg:4326'}
police_df_Austin = gpd.read_file('../input/data-science-for-good/cpe-data/'
                                 +'Dept_37-00027/37-00027_Shapefiles/APD_DIST.shp')
police_df_Austin.crs = {'init' :'esri:102739'}
police_df_Austin = police_df_Austin.to_crs(epsg='4326')
force_clean_df.head()


# In[ ]:


locations_df = pd.DataFrame()
locationlist=[]
locations_df['LOCATION_LONGITUDE']=force_clean_df['LOCATION_LONGITUDE'].astype(float)
locations_df['LOCATION_LATITUDE'] =force_clean_df['LOCATION_LATITUDE'].astype(float)
for i, r in locations_df.iterrows():
    locationlist.append([r['LOCATION_LONGITUDE'],r['LOCATION_LATITUDE']])


# 

# In[ ]:


fig1,ax = plt.subplots(1,2,figsize=(20,10))
police_df_Austin.plot(ax=ax[0],column='SECTOR',alpha=0.5,legend=True)
s=force_clean_df['INCIDENT_REASON']
s.value_counts().plot(kind='bar',ax=ax[1],rot=10)
force_clean_df.plot(marker='.',ax=ax[0])


# From the arrest plot we can see that for sector APT it only has one record. Due to the number of arrest  record in this sector, it most likely will bring bias to our result, so we will exclude this department.
# Then let have a look at the race ratio among all the arrest record
# subject race pie chart

# In[ ]:



force_clean_df.SUBJECT_RACE.value_counts()
print(force_clean_df.SUBJECT_RACE.value_counts())
fig1, ax1 = plt.subplots()
ax1.pie(force_clean_df.SUBJECT_RACE.value_counts(),labels=force_clean_df.SUBJECT_RACE.value_counts().keys(),autopct='%1.1f%%',startangle=90,  shadow=True)
ax1.axis('equal')
plt.show()


# The pie chart shows that Hispanic, White, and Black construct most of the area, so if other departments also has a large population of this three races, then we could only include the arrest ratio of these there major races in our frequent pattern mining procedure.
# 
# 
# Texas Census file:

# In[ ]:


census_poverty_df = pd.read_csv('../input/data-science-for-good/cpe-data/'+
                                'Dept_37-00027/37-00027_ACS_data/37-00027_ACS_poverty/'+
                                'ACS_15_5YR_S1701_with_ann.csv')
census_poverty_df = census_poverty_df.iloc[1:].reset_index(drop=True)
census_poverty_df = census_poverty_df.rename(columns={'GEO.id2':'GEOID'})
census_tracts_gdf = gpd.read_file("../input/texgeo/cb_2017_48_tract_500k /cb_2017_48_tract_500k//cb_2017_48_tract_500k.shp")
census_merged_gdf = census_tracts_gdf.merge(census_poverty_df, on = 'GEOID')
census_merged_gdf = census_merged_gdf.to_crs(epsg='4326')
census_merged_gdf.head()


# Inorder to have a more clear view of the arrest data we plot the data on the map

# In[ ]:


mapa = folium.Map([30.3, -97.7],zoom_start=10, height=500)
locations_df = force_clean_df[["LOCATION_LATITUDE", "LOCATION_LONGITUDE"]].copy()
locations_df = locations_df.iloc[locations_df[['LOCATION_LATITUDE','LOCATION_LONGITUDE']].dropna().index].reset_index(drop=True)
locations_df["LOCATION_LATITUDE"] = locations_df["LOCATION_LATITUDE"].astype('float')
locations_df["LOCATION_LONGITUDE"] = locations_df["LOCATION_LONGITUDE"].astype('float')
locationlist = locations_df.values.tolist()[-2000:]
for point in range(0, len(locationlist)):
    folium.CircleMarker(locationlist[point], radius=0.1, color='red').add_to(mapa)

mapa


# District list:

# In[ ]:


DB_district_list=[k for k in force_clean_df['LOCATION_DISTRICT'].value_counts().keys()]
DB_district_list


# Update the arrest file GeoID by check if it is contianed by the cunsus track file. 

# The final file for index calculation:

# In[ ]:


final_aus_arrest=pd.read_csv('../input/aus-final/Dept_37-00027_arrest_GEo.csv')
final_aus_arrest.head()


# # Minneapolis

# Minneapolis arrest file

# In[ ]:


force_df = pd.read_csv("../input/data-science-for-good/cpe-data/Dept_24-00013/24-00013_UOF_2008-2017_prepped.csv")
force_clean_df = force_df.loc[1:].reset_index(drop=True)
force_clean_df ['LOCATION_LONGITUDE']= pd.to_numeric(force_clean_df['LOCATION_LONGITUDE'], downcast='float') 
force_clean_df ['LOCATION_LATITUDE']= pd.to_numeric(force_clean_df['LOCATION_LATITUDE'], downcast='float') 
force_clean_df = force_clean_df[np.isfinite(force_clean_df['LOCATION_LONGITUDE'])]
force_clean_df=force_clean_df[force_clean_df['LOCATION_LONGITUDE']!=0].reset_index(drop=True)
foo=lambda x: Point(x['LOCATION_LONGITUDE'],x['LOCATION_LATITUDE'])
force_clean_df['geometry'] = (force_clean_df.apply(foo, axis=1))
force_clean_df = gpd.GeoDataFrame(force_clean_df, geometry='geometry')
force_clean_df.crs = {'init' :'epsg:4326'}
force_clean_df.head()


# Minneapolis police department file

# In[ ]:


police_df = gpd.read_file( '../input/data-science-for-good/cpe-data//Dept_24-00013/'+
                '24-00013_Shapefiles/Minneapolis_Police_Precincts.shp')
police_df.head()


# Arrest and police shape plot

# In[ ]:


fig1,ax = plt.subplots(1,2,figsize=(20,10))
police_df.plot(ax=ax[0],alpha=0.5,legend=True)
s=force_clean_df['REASON_FOR_FORCE']
s.value_counts().plot(kind='bar',ax=ax[1])
force_clean_df.plot(marker='.',ax=ax[0],column='LOCATION_DISTRICT',legend=True)


# Subject race pie chart

# In[ ]:


force_clean_df.SUBJECT_RACE.value_counts()
print(force_clean_df.SUBJECT_RACE.value_counts())
fig1, ax1 = plt.subplots()
ax1.pie(force_clean_df.SUBJECT_RACE.value_counts(),labels=force_clean_df.SUBJECT_RACE.value_counts().keys(),autopct='%1.1f%%',startangle=90,  shadow=True)
ax1.axis('equal')
plt.show()


# The pie chart shows that the major races are white and black in this area
# 

# Census data  file

# In[ ]:


census_tract_df=gpd.read_file("../input/minneapolis/cb_2017_27_tract_500k /cb_2017_27_tract_500k.shp")
census_tract_df.head()


# Arrest record on the map

# In[ ]:


mapa = folium.Map([45, -93.3], height=500, zoom_start=11)

folium.GeoJson(police_df).add_to(mapa)
locations_df = force_clean_df[["LOCATION_LATITUDE", "LOCATION_LONGITUDE"]].copy()
notna = locations_df[['LOCATION_LATITUDE','LOCATION_LONGITUDE']].dropna().index
locations_df = locations_df.iloc[notna].reset_index(drop=True)
locations_df["LOCATION_LATITUDE"] = locations_df["LOCATION_LATITUDE"].astype('float')
locations_df["LOCATION_LONGITUDE"] = locations_df["LOCATION_LONGITUDE"].astype('float')
locationlist = locations_df.values.tolist()[-2000:]
for point in range(0, len(locationlist)):
    folium.CircleMarker(locationlist[point], radius=0.1, color='red').add_to(mapa)

mapa 


# In[ ]:


overlap_police=gpd.GeoDataFrame(columns=census_tract_df.columns)
item_set=[]
for index1,x in police_df.iterrows():
    lst_geoid=[]
    for index2, y in census_tract_df.iterrows():
        if x['geometry'].contains(y['geometry']) or y['geometry'].intersects(x['geometry']) or y['geometry'].contains(x['geometry']):
            if y['GEOID'] not in item_set:
                lst_geoid.append(y['GEOID'])
                item_set.append(y['GEOID'])
                police_df.at[index1,'GEOid']=lst_geoid
                overlap_police.loc[-1]=y
                overlap_police.index = overlap_police.index + 1


# Merged census data with the arrest record:

# In[ ]:


fig2,ax2 = plt.subplots()
force_clean_df.plot(ax=ax2,marker='.',column='LOCATION_DISTRICT',legend=True,markersize=20)
overlap_police.plot(ax=ax2,color='0.7',alpha=.5,edgecolor='white')

fig2.set_size_inches(10,10)


# After the data set merged, then we can assign each GEOid to the arrest record by check if the arrest happened location is in that census tract

# The final file for index calculation after update the GEOid for each arrest record:

# In[ ]:


final_aus_arrest=pd.read_csv('../input/min-fin/Dept_2400013_arrest_GEo.csv')
final_aus_arrest.head()


# # Saint Paul

# Arrest repord with shapely point

# In[ ]:


force_df = pd.read_csv("../input/data-science-for-good/cpe-data/Dept_24-00098/24-00098_Vehicle-Stops-data.csv")
force_clean_df = force_df.loc[1:].reset_index(drop=True)
force_clean_df ['LOCATION_LONGITUDE']= pd.to_numeric(force_clean_df['LOCATION_LONGITUDE'], downcast='float') 
force_clean_df ['LOCATION_LATITUDE']= pd.to_numeric(force_clean_df['LOCATION_LATITUDE'], downcast='float') 
force_clean_df = force_clean_df[np.isfinite(force_clean_df['LOCATION_LONGITUDE'])]
force_clean_df=force_clean_df[force_clean_df['LOCATION_LONGITUDE']!=0].reset_index(drop=True)
foo=lambda x: Point(x['LOCATION_LONGITUDE'],x['LOCATION_LATITUDE'])
force_clean_df['geometry'] = (force_clean_df.apply(foo, axis=1))
force_clean_df = gpd.GeoDataFrame(force_clean_df, geometry='geometry')
force_clean_df.crs = {'init' :'epsg:4326'}
force_clean_df.head()


# Police department geometry data file

# In[ ]:


police_df = gpd.read_file('../input/data-science-for-good/cpe-data/Dept_24-00098/24-00098_Shapefiles/StPaul_geo_export_6646246d-0f26-48c5-a924-f5a99bb51c47.shp')
police_df.head()


# The St Paul dataset contains much more arrest records than other data set and we can also see this by plot the overlapping plot of ploce department and arrest record

# In[ ]:



fig2,ax2 = plt.subplots()
force_clean_df.plot(ax=ax2)
police_df.plot(ax=ax2,color='0.7',alpha=.5,edgecolor='white')

fig2.set_size_inches(10,10)


# Subject Race pie Chart:

# In[ ]:


force_clean_df.SUBJECT_RACE.value_counts()
print(force_clean_df.SUBJECT_RACE.value_counts())
fig1, ax1 = plt.subplots()
ax1.pie(force_clean_df.SUBJECT_RACE.value_counts(),labels=force_clean_df.SUBJECT_RACE.value_counts().keys(),autopct='%1.1f%%',startangle=90,  shadow=True)
ax1.axis('equal')
plt.show()


# From the pie chart we can also point out that the major races are White and Black

# Census tract data plot:

# In[ ]:


census_tract_df=gpd.read_file("../input/stpaul/cb_2015_27_tract_500k/cb_2015_27_tract_500k.shp")
census_tract_df.plot()


# we can merge the census tract data set with arrest reocrd and update the GEOid to each record like we have down for the previous two dataset

# The final data set for index calculation and mining:

# In[ ]:


final_arrest=pd.read_csv('../input/stpa-final/Dept_2400098_arrest_GEo.csv')
final_arrest.head()


# # Dallas

# Arrest record with shapely points

# In[ ]:


force_df = pd.read_csv("../input/data-science-for-good/cpe-data/Dept_37-00049/37-00049_UOF-P_2016_prepped.csv")
force_clean_df = force_df.loc[1:].reset_index(drop=True)
force_clean_df ['LOCATION_LONGITUDE']= pd.to_numeric(force_clean_df['LOCATION_LONGITUDE'], downcast='float') 
force_clean_df ['LOCATION_LATITUDE']= pd.to_numeric(force_clean_df['LOCATION_LATITUDE'], downcast='float') 
force_clean_df = force_clean_df[np.isfinite(force_clean_df['LOCATION_LONGITUDE'])]
force_clean_df=force_clean_df[force_clean_df['LOCATION_LONGITUDE']!=0].reset_index(drop=True)
foo=lambda x: Point(x['LOCATION_LONGITUDE'],x['LOCATION_LATITUDE'])
force_clean_df['geometry'] = (force_clean_df.apply(foo, axis=1))
force_clean_df = gpd.GeoDataFrame(force_clean_df, geometry='geometry')
force_clean_df.crs = {'init' :'epsg:4326'}
force_clean_df.head()


# police department shape file

# In[ ]:


police_df = gpd.read_file('../input/data-science-for-good/cpe-data/Dept_37-00049/37-00049_Shapefiles/EPIC.shp')
police_df=police_df.to_crs(epsg='4236')
police_df.head()


# Map the arrest record and police dpartment file on the plot

# In[ ]:


fig2,ax2 = plt.subplots()
force_clean_df.plot(ax=ax2)
police_df.plot(ax=ax2,color='0.7',alpha=.5,edgecolor='white')

fig2.set_size_inches(10,10)


# Subject race pie chart

# In[ ]:


force_clean_df.SUBJECT_RACE.value_counts()
print(force_clean_df.SUBJECT_RACE.value_counts())
fig1, ax1 = plt.subplots()
ax1.pie(force_clean_df.SUBJECT_RACE.value_counts(),labels=force_clean_df.SUBJECT_RACE.value_counts().keys(),autopct='%1.1f%%',startangle=90,  shadow=True)
ax1.axis('equal')
plt.show()


# The pie chart indicate the major races are black,white, and Hispanic

# In[ ]:


mapa = folium.Map([32.78, -96.79],zoom_start=10, height=500)
locations_df = force_clean_df[["LOCATION_LATITUDE", "LOCATION_LONGITUDE"]].copy()
locations_df = locations_df.iloc[locations_df[['LOCATION_LATITUDE','LOCATION_LONGITUDE']].dropna().index].reset_index(drop=True)
locations_df["LOCATION_LATITUDE"] = locations_df["LOCATION_LATITUDE"].astype('float')
locations_df["LOCATION_LONGITUDE"] = locations_df["LOCATION_LONGITUDE"].astype('float')
locationlist = locations_df.values.tolist()[-2000:]
for point in range(0, len(locationlist)):
    folium.CircleMarker(locationlist[point], radius=0.1, color='red').add_to(mapa)

mapa


# Census data plot

# In[ ]:


census_tract_df=gpd.read_file("../input/dallas/cb_2017_48_tract_500k /cb_2017_48_tract_500k/cb_2017_48_tract_500k.shp")
census_tract_df.plot()


# After check for overlap and updated the GEOid for each arrest record we now have the final data:

# In[ ]:


final_arrest=pd.read_csv('../input/dala-fin/Dept_3700049_arrest_GEo.csv')
final_arrest.head()


# # Orlando

# Arrest record data set:

# In[ ]:


force_df = pd.read_csv("../input/data-science-for-good/cpe-data/Dept_35-00016/35-00016_UOF-OIS-P.csv")
force_clean_df = force_df.loc[1:].reset_index(drop=True)
force_clean_df ['LOCATION_LONGITUDE']= pd.to_numeric(force_clean_df['LOCATION_LONGITUDE'], downcast='float') 
force_clean_df ['LOCATION_LATITUDE']= pd.to_numeric(force_clean_df['LOCATION_LATITUDE'], downcast='float') 
force_clean_df = force_clean_df[np.isfinite(force_clean_df['LOCATION_LONGITUDE'])]
force_clean_df=force_clean_df[force_clean_df['LOCATION_LONGITUDE']!=0].reset_index(drop=True)
foo=lambda x: Point(x['LOCATION_LONGITUDE'],x['LOCATION_LATITUDE'])
force_clean_df['geometry'] = (force_clean_df.apply(foo, axis=1))
force_clean_df = gpd.GeoDataFrame(force_clean_df, geometry='geometry')
force_clean_df.crs = {'init' :'epsg:4326'}
force_clean_df.head()


# Police department sectors file:

# In[ ]:


police_df = gpd.read_file('../input/data-science-for-good/cpe-data/Dept_35-00016/35-00016_Shapefiles/OrlandoPoliceSectors.shp')
police_df=police_df.to_crs(epsg='4236')
police_df.head()


# Cobining plot of police department sectors and arrest records

# In[ ]:


fig2,ax2 = plt.subplots()
force_clean_df.plot(ax=ax2)
police_df.plot(ax=ax2,color='0.7',alpha=.5,edgecolor='white')

fig2.set_size_inches(10,10)


# Subject race among all the arrest record

# Subject Race Pie Chart:

# In[ ]:


force_clean_df.SUBJECT_RACE.value_counts()
print(force_clean_df.SUBJECT_RACE.value_counts())
fig1, ax1 = plt.subplots()
ax1.pie(force_clean_df.SUBJECT_RACE.value_counts(),labels=force_clean_df.SUBJECT_RACE.value_counts().keys(),autopct='%1.1f%%',startangle=90,  shadow=True)
ax1.axis('equal')
plt.show()


# The pie chart indicate the major races are Black and White among arrest records

# Arrest data in the real map

# In[ ]:


mapa = folium.Map([28.53, -81.39],zoom_start=10, height=500)
locations_df = force_clean_df[["LOCATION_LATITUDE", "LOCATION_LONGITUDE"]].copy()
locations_df = locations_df.iloc[locations_df[['LOCATION_LATITUDE','LOCATION_LONGITUDE']].dropna().index].reset_index(drop=True)
locations_df["LOCATION_LATITUDE"] = locations_df["LOCATION_LATITUDE"].astype('float')
locations_df["LOCATION_LONGITUDE"] = locations_df["LOCATION_LONGITUDE"].astype('float')
locationlist = locations_df.values.tolist()[-2000:]
for point in range(0, len(locationlist)):
    folium.CircleMarker(locationlist[point], radius=0.1, color='red').add_to(mapa)

mapa


# Census geometry data plot : 

# In[ ]:


census_tract_df=gpd.read_file("../input/orlando/cb_2016_12_tract_500k/cb_2016_12_tract_500k.shp")
census_tract_df.plot()


# After check for overlap and updated the GEOid for each arrest record we now have the final data:

# In[ ]:


final_arrest=pd.read_csv('../input/orlan-final/Dept_35-00016_arrest_GEo.csv')
final_arrest.head()


# **For the check of loverlap and geometry relationship process, I did not include those functions and codes in the kernal due to the runtime of each department check

# By the previous step, the data which laied out of the department range are excluded and also by analysing the subject race of each department, we will only include Black and White tow major races to calculate the arrest ratio and index ratio factors in the future.
# After preprocessing the data, the next step is to calculate the Racism Index and do the frequent pattern mining

# Contriburion:
# The work is contribute by Shuaidong Pan, Faner Lin and Weijian Li

# In[ ]:




