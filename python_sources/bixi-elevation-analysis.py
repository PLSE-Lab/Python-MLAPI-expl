#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd 
import numpy
import matplotlib.pyplot as plt
import geopy.distance
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point


# # 1) Questions
# How does the elevation of bixi stations influence trips and how does the date/time come into play?
# 

# # Import Code

# In[ ]:


OD_2017 = pd.read_csv('../input/OD_2017.csv', low_memory=False, index_col=0);
Stations_2017 = pd.read_csv('../input/Stations_2017.csv', low_memory=False);
Elevation = pd.read_csv('../input/POINT_DATA.csv', low_memory=False);
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# # Check the shape of the Data

# In[ ]:


OD_2017.head()


# In[ ]:


Stations_2017.head()


# In[ ]:


Elevation.head()


# ## Rename columns

# In[ ]:


Elevation.columns =['longitude','latitude','elevation']
Elevation.head()


# In[ ]:


#Add column
Stations_2017['elevation'] = 'e'


# In[ ]:


#Work with copy of data
Stations_2017_Elevations= Stations_2017.copy()
Stations_2017_Elevations.head()


# # Combining the data from the elevation coordinates with bixi station coordinates
# 
# The coordinates elevation data doesn't match the coordinates in the bixi data exactly so we take the closest data point and asociate it with that one

# In[ ]:


for index, row in Stations_2017.iterrows(): 
    latitude = str(row['latitude'])
    longitude = str(row['longitude'])
    query = Elevation.query('longitude < '+longitude+' + 0.0001 & longitude > '+longitude+' - 0.0001 & latitude < '+latitude+' + 0.0001 & latitude > '+latitude+' - 0.0001')
   
    rows, cols = query.shape
    if(rows ==1):
        Stations_2017.loc[index, 'elevation'] = query['elevation'].iloc[0]


# In[ ]:


Stations_2017.head(10)


# In[ ]:


Stations_2017.query('elevation == "e"')


# # Missing Values
# As we can see there are multiple missing values. I will therefore get these elevations manually by using https://www.freemaptools.com/elevation-finder.htm

# In[ ]:


Stations_2017.to_csv(path_or_buf='Stations_2017_elevations.csv')


# In[ ]:


Stations_2017_Elevations= pd.read_csv('../input/Stations_2017_elevations_with_manually_added_missing_elevations.csv', low_memory=False,encoding = "cp1252");


# In[ ]:


#new Elevations with missing values added
Stations_2017_Elevations.head(100)


# # Check if there are any missing values

# In[ ]:


Stations_2017_Elevations.isnull().sum()


# # Analysis

# In[ ]:


plt.figure(figsize=(15,5));
plt.subplot(1,2,1);
Stations_2017_Elevations['elevation'].plot.hist(bins=40);
plt.xlabel('Elevation');


# # Merge Station data with bixi trip data

# In[ ]:


df_complete = OD_2017.sort_values(by = ['start_station_code'])
Stations_2017_Elevations.sort_values(by = ['code'])

df_complete = pd.merge(df_complete, Stations_2017_Elevations,  how='left', left_on = 'start_station_code', right_on = 'code')


# In[ ]:


df_complete.head(100)


# # Cleaning the Code
# Renaming columns and removing redundant columns

# In[ ]:


df_complete.rename(columns={'latitude': 'latitude_start', 'longitude': 'longitude_start','elevation': 'elevation_start','name': 'name_start', 'is_public': 'is_public_start'}, inplace=True)


# In[ ]:


df_complete.drop(columns=['code'],axis=1, inplace=True)
df_complete.head()


# In[ ]:


df_complete.loc[df_complete['end_station_code'] == 'Tabletop (RMA)']


# In[ ]:


# Remove data that doesn't match rest of data
df_complete = df_complete[df_complete.end_station_code !='Tabletop (RMA)']


# In[ ]:


#cast data to number
df_complete["end_station_code"] = pd.to_numeric(df_complete["end_station_code"]) 


# In[ ]:


df_complete_end_elevation = pd.merge(df_complete, Stations_2017_Elevations,  how='left', left_on = 'end_station_code', right_on = 'code')


# In[ ]:


df_complete_end_elevation.head()


# ## Rename end station longitudes 

# In[ ]:


df_complete_end_elevation.rename(columns={'latitude': 'latitude_end', 'longitude': 'longitude_end','name': 'name_end','elevation': 'elevation_end','name': 'name_end', 'is_public': 'is_public_end'}, inplace=True)


# In[ ]:


df_complete_end_elevation.drop(columns=['code'],axis=1, inplace=True)
df_complete_end_elevation.head()


# 
# # Create Columns For Distance Traveled and Elevation Variation
# 
# 

# In[ ]:


df_complete_end_elevation['elevation_difference'] = df_complete_end_elevation.apply(lambda row: row.elevation_end - row.elevation_start, axis=1)
df_complete_end_elevation.head(5)


# In[ ]:


#df_complete_end_elevation['distance_between_stations'] = df_complete_end_elevation.apply(lambda row: geopy.distance.distance((row.latitude_start,row.longitude_start), (row.latitude_end ,row.longitude_end)), axis=1)
df_complete_end_elevation.shape


# In[ ]:


row = df_complete_end_elevation.iloc[1]
distance = geopy.distance.distance((row.latitude_start,row.longitude_start), (row.latitude_end ,row.longitude_end))
print(distance)


# In[ ]:


df_distances = pd.DataFrame(columns=['distance'])
df =df_complete_end_elevation.iloc[0:5]
df


# # Calculate Elevation Distance between start and end station

# In[ ]:


#Takes 40 minutes to calculate on my laptop, no need to run as dataset with distance values are loaded later on

#for index, row in df_complete_end_elevation.iterrows():
#    df_complete_end_elevation.set_value(index,'distance',geopy.distance.distance((row.latitude_start,row.longitude_start), (row.latitude_end ,row.longitude_end)))
#    if(index%10000==0):
#        print(index)
#        
#df_complete_end_elevation.head(10)


# In[ ]:


#df_complete_end_elevation.to_csv('complete_data_elevations_and_distance.csv')


# # Full Data Set Loaded from previous run

# In[ ]:


df_bixi = pd.read_csv('../input/complete_data_elevations_and_distance.csv', low_memory=False);


# In[ ]:


df_bixi.info()


# In[ ]:


# change object distance with end km to float
df_bixi['distance'] = df_bixi['distance'].astype(str).str[:-3].astype(float)


# In[ ]:


df_bixi.info()


# In[ ]:



df_bixi.head()


# # Analysis

# In[ ]:


plt.figure(figsize=(20,10));
plt.subplot(1,2,1);
df_bixi['distance'].plot.hist(bins=40)
plt.xlabel('Distance Km');


plt.subplot(1,2,2);
df_bixi['elevation_difference'].plot.hist(bins=100)
plt.xlabel('Elevation Difference (m)');


# In[ ]:



fig = plt.figure(figsize=(10,10));
ax = fig.gca()
ax.set_xticks(numpy.arange(0, 30, 2))
ax.set_yticks(numpy.arange(-50, 50, 2))
plt.grid()
plt.boxplot(df_bixi['elevation_difference'], 0, '')


# In[ ]:


df_bixi.describe()


# In[ ]:


df_bixi.median()


# In[ ]:


df_bixi[df_bixi['elevation_difference'] <0 ].count()


# In[ ]:


#show data about elevation on specific days and times
df_bixi['start_date'] = pd.to_datetime(OD_2017['start_date'])
df_bixi['end_date'] = pd.to_datetime(OD_2017['end_date'])
df_bixi[['start_date', 'end_date']].dtypes


# In[ ]:


plt.figure(figsize=(15,5));
df_bixi['duration_sec'].plot.hist(bins=100);
plt.xlabel('Duration');


# In[ ]:




plt.subplot(1,2,1);
df_bixi.groupby('is_member').mean()['elevation_difference'].plot(kind='bar', color='#1f77b4');
plt.title('Mean Elevation Difference');


# In[ ]:


df_bixi['weekday'] = df_bixi.start_date.dt.dayofweek
df_bixi['hour'] = df_bixi.start_date.dt.hour
df_bixi['month'] = df_bixi.start_date.dt.month
df_bixi['daynum'] = df_bixi.start_date.dt.dayofyear


# In[ ]:


plt.figure(figsize=(15,5));
plt.subplot(1,3,1);
df_bixi.groupby('weekday').count()['duration_sec'].plot(kind='bar', color='#1f77b4');

plt.subplot(1,3,2);
df_bixi.groupby('hour').count()['duration_sec'].plot(kind='bar', color='#1f77b4');

plt.subplot(1,3,3);
df_bixi.groupby('month').count()['duration_sec'].plot(kind='bar', color='#1f77b4');


# In[ ]:


dfp = df_bixi.pivot_table(columns='hour',index='weekday', aggfunc=np.mean)['elevation_difference'];
plt.figure(figsize=(18,5));
plt.title('Pivot table: Mean Elevation Difference');
plt.imshow(dfp,interpolation='none');
hours = range(24);
hourlabels = map(lambda x: str(x)+'h',hours);
days = range(7);
daylabels = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
plt.xticks(hours,hourlabels,rotation=90);
plt.yticks(days,daylabels);
plt.colorbar();


# # Analysis:
# It seems that on Friday afternoons and Saturday nights, when people go out, they take bixis downhill. Most likely from the plateau down. 
# The term Lazy Sunday seems to fit well too as all day the average is around -4 meters in elevation.

# In[ ]:


dfp = df_bixi.pivot_table(columns='hour',index='weekday', aggfunc=np.mean)['distance'];
plt.figure(figsize=(18,5));
plt.title('Pivot table: Trip Distance (km)');
plt.imshow(dfp,interpolation='none');
hours = range(24);
hourlabels = map(lambda x: str(x)+'h',hours);
days = range(7);
daylabels = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
plt.xticks(hours,hourlabels,rotation=90);
plt.yticks(days,daylabels);
plt.colorbar();


# In[ ]:


Stations_2017_Elevations['Coordinates']  = list(zip(Stations_2017_Elevations.longitude, Stations_2017_Elevations.latitude))


# In[ ]:


Stations_2017_Elevations['Coordinates'] = Stations_2017_Elevations['Coordinates'].apply(Point)


# In[ ]:


gdf = gpd.GeoDataFrame(Stations_2017_Elevations, geometry='Coordinates')
print(gdf.head())
gdf.plot()


# In[ ]:


vmin, vmax = -10, 100

ax = gdf.plot(column='elevation', colormap='hot', vmin=vmin, vmax=vmax)

# add colorbar
fig = ax.get_figure()
cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
sm = plt.cm.ScalarMappable(cmap='hot', norm=plt.Normalize(vmin=vmin, vmax=vmax))
# fake up the array of the scalar mappable. Urgh...
sm._A = []
fig.colorbar(sm, cax=cax)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




