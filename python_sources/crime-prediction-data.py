#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd

raw_data = pd.read_csv("../input/nypd-complaint-data-historic/rows.csv",
                      usecols=["CMPLNT_FR_DT", "Latitude", "Longitude"]).dropna()

# Data is supposed to start 2006-01-01, filter previous dates
day_one = np.datetime64('2006-01-01')
days = (pd.to_datetime(raw_data['CMPLNT_FR_DT'], 
                 format='%m/%d/%Y', errors='coerce') - day_one).dt.days
days = days[days.isnull() == False]
days = days[days > 0]

# Filter outliers
rd = raw_data
rd = rd[rd['Latitude'] > rd['Latitude'].mean() - 3 * rd['Latitude'].std()]                [rd['Latitude'] < rd['Latitude'].mean() + 3 * rd['Latitude'].std()]
rd = rd[rd['Longitude'] > rd['Longitude'].mean() - 3 * rd['Longitude'].std()]                [rd['Longitude'] < rd['Longitude'].mean() + 3 * rd['Longitude'].std()]

# Discretize latitude and longitude
min_lat, max_lat = min(rd["Latitude"]), max(rd["Latitude"])
min_lng, max_lng = min(rd["Longitude"]), max(rd["Longitude"])
GRID_LNG_DIM = 100
GRID_LAT_DIM = 120
lng_step = (max_lng - min_lng) / (GRID_LNG_DIM - 1)
lat_step = (max_lat - min_lat) / (GRID_LAT_DIM - 1)
def to_grid_lat(lat):
    return (lat - min_lat) // lat_step
def to_grid_lng(lng):
    return (lng - min_lng) // lng_step

# Create final dataset
data = pd.DataFrame(rd["CMPLNT_FR_DT"])
data["Days"] = days
data["GridLat"] = to_grid_lat(rd['Latitude'])
data["GridLng"] = to_grid_lng(rd['Longitude'])
data = data.dropna()
data["Days"] = data["Days"].astype("int16")
data["GridLat"] = data["GridLat"].astype("int8")
data["GridLng"] = data["GridLng"].astype("int8")
# To split the validation set to the most recent crimes
data.sort_values(by=["Days"])


# In[ ]:


# Histogram of crimes by date
data["Days"].hist(bins=100)


# In[ ]:


import seaborn

# Criminality in the map
seaborn.jointplot(x="GridLng", y="GridLat", data=data, kind="hex")


# In[ ]:


# We create a (4748, 101, 121) boolean array to track if in a specific day and place there was a crime
crime_grid = np.zeros((max(data["Days"]) + 1, GRID_LNG_DIM + 1, GRID_LAT_DIM + 1), dtype="bool")
crime_grid[data["Days"], data["GridLng"], data["GridLat"]] = True


# In[ ]:


# Proportion of criminality
len(crime_grid[crime_grid == True]) / crime_grid.size


# In[ ]:


# For each slot in crime_grid, we take a 27-array (3x3x3) neighbours in each dimension

import itertools
# index = itertools.product(range(-1,2), repeat=3)
index = itertools.product(range(-3,0), range(-1,2), range(-1,2))
index = np.array(list(index))

# arrays_27 = np.stack((np.roll(crime_grid, ind) for ind in index), axis=-1)[1:-1, 1:-1, 1:-1]
arrays_27 = np.stack((np.roll(crime_grid, ind) for ind in index), axis=-1)[3:, 1:-1, 1:-1]


# In[ ]:


arrays_27.shape


# In[ ]:


y_label = crime_grid[3:, 1:-1, 1:-1]


# In[ ]:


y_label.shape


# In[ ]:


import h5py

with h5py.File("nypd_crime_data.hdf5", "w") as f:
    f.create_dataset("X", data=arrays_27, dtype='bool', compression="gzip", compression_opts=9)
    f.create_dataset("y", data=y_label, dtype='bool', compression="gzip", compression_opts=9)


# In[ ]:


with h5py.File("nypd_crime_data_date_lat_lng.hdf5", "w") as f:
    f.create_dataset("X", data=data[["Days", "GridLat", "GridLng"]], compression="gzip", compression_opts=9)

