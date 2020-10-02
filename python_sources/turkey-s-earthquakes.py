#!/usr/bin/env python
# coding: utf-8

# **The purpose of this kernel:**
# * Learning basic pandas methods (importing data from csv file, getting familar with head, tail, column, type, value_counts, info, to_datetime, set_index, loc, resample methods)
# * Learning filtering methods with Pandas and Numpy (logical_and method)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("../input/database.csv")


# In[ ]:


# First 10 entries
data.head(10)


# In[ ]:


# Last 10 entries
data.tail(10)


# In[ ]:


data.columns


# In[ ]:


# Making data columns' name lowercase
data.columns = [i.lower() for i in data.columns]


# In[ ]:


# Removing spaces in column names
# There are names with one and two spaces. So we can create this algorithm.
data.columns = [i.split()[0]+'_'+i.split()[1] if len(i.split())==2 
                else i.split()[0]+'_'+i.split()[1]+'_'+i.split()[2] if len(i.split())==3
                else i 
                for i in data.columns]


# In[ ]:


# All column names are lowercase and without spaces
data.head(1)


# In[ ]:


# Cause of the seismic activity
print(data.type.value_counts(dropna=False))


# In[ ]:


# Source of the seismic data
print(data.source.value_counts(dropna=False))


# In[ ]:


data.info()


# **Data null / non-null objects**
# * 18951 null value for Depth Error
# * 16315 null value for Depth Seismic Stations  
# * 3 null value for Magnitude Type
# * 23082 null value for Magnitude Error
# * 20848 null value for Magnitude Seismic Stations
# * 16113 null value for Azimuthal Gap
# * 21808 null value for Horizontal Distance
# * 22256 null value for Horizontal Error
# * 6060 nul value for Root Mean Square

# In[ ]:


# Converting 'date' column to datetime64[ns] type. So we can use this column in time series
data.date = pd.to_datetime(data.date)


# * Turkey Coordinates : 
# *     Latitude (northing) between 36 and 42 degrees
# *     Longitude (easting) between 26 and 45 degrees
# * We can use these coordinates to filter the earthquake data, so we can create a new dataframe of earthquakes that occured in Turkey.
# 
#         

# In[ ]:


# Creating a new dataframe Earthquakes of Turkey
Lat = np.logical_and(data.latitude>=36, data.latitude<=42)
Long = np.logical_and(data.longitude>=26, data.longitude<=45)
dataTR = data[Lat & Long]
# Basicaly we're filtering the data by setting a rectangle polygon.

# Saving .csv file to computer
# dataTR.to_csv("....../dataTR.csv")


# In[ ]:


# Viewing the new data frame.
dataTR
# Because of the filtering index numbers (73, 175, 241...) are disordered. If we're going to use ID we can correct this
# index numbers.
# I'd like to use this dataTR for time series so I'll convert the index to date.


# In[ ]:


# Changing the index to date
dataTR = dataTR.set_index('date')
dataTR


# In[ ]:


# Filtering the earthquake data for 1999 year
#dataTR.loc['1999']
# Filtering the earthquake data for 1999 and 2000 years
dataTR.loc['1999':'2000']


# In[ ]:


# Viewing 'latitude','longitude','type','depth','magnitude' columns
dataTRsimple = dataTR[['latitude','longitude','type','depth','magnitude']]
dataTRsimple.head()


# In[ ]:


# Creating a function which divides the depth value by 2. (For study purpose. It doesn't make sense.)
def divide(x):
    return x/2
dataTRsimple.depth = dataTRsimple.depth.apply(divide)
#dataTRsimple.depth = dataTRsimple.depth.apply(lambda x: x/2)  # (2nd method: lambda function)
dataTRsimple.head()


# In[ ]:


# To learn the data's index
print(dataTRsimple.index.name)


# In[ ]:


# We can change index name 'date' to something else.
dataTRsimple.index.name = 'time_id'
dataTRsimple.head()


# In[ ]:


# Or we can create new column for ID
#dataTRsimple.info()  # We have total of 123 entries for this data. So we can create ID's from 1 to 123.

# If we set the ID's as a new index we will lose the date(time_id) values.
# To prevent this we should create a new column for date values
dataTRsimple['date'] = dataTRsimple.index  # Creating a new column called 'date' and copying the index to this column.
dataTRsimple['ID'] = range(1,124)  # Creating a new column for ID's
dataTRsimple = dataTRsimple.set_index('ID')  # Set the index as ID
dataTRsimple.head()


# In[ ]:


# Hierarchical indexing

# We can set two index for data.
dataTRsimple['ID'] = dataTRsimple.index  # Creating a new column 'ID' and copying our index to this column
dataTRsimple = dataTRsimple.set_index(["ID","date"])  # Setting two index. ID is outer index, date is inner index
dataTRsimple.head()


# In[ ]:


# Calculating mean value for all columns according to years ('A')
#dataTR.resample('A').mean() 
# Calculating mean value for all columns according to months ('M')
#dataTR.resample('M').mean()
# Linear interpolation
#dataTR.resample('M').first().interpolate('linear')  # This method can be used for filling empty values between first and last data

# These calculations are for study purposes. Don't make any sense for this data.


# **Summary**
# 1. Filtered the data in order the get Turkey's earthquakes by using latitude & longtitude values (numpy logical_and method.) and created a new dataframe from this filtered data.
# 2. Changed the data index to date and filtered the new dataframe by using .loc slicing method to view the Turkey's earthquakes in 1999 and 2000. There were two big earthquakes in 1999 (17 August Izmit and 12 November Duzce). We can also see some aftershocks which are very close to the epicenter of major earthquakes. (Unfortunately smaller than Mw 5.5 earthquake data is not avaible in this dataset)
# 3. It's possible to make custom filters and create new dataframes from the main data as csv files. Later these csv files can be used in GIS programs like QGIS to add a new layer to maps. (Maybe filtering can be done QGIS platform?)
# 4. Learned manipulating data's index.
# 
