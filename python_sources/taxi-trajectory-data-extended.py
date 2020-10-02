#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This kernel analyses and processes Taxi Trajectory Data and add the following fields:
# 
#     TRIP_DISTANCE: (float): It contains the total geodesic distance calculated from all consecutive pairs of coordinates.
# 
#     TRIP_TIME: (float): It contains the total time calculated from the number of pairs of coordinates, knowing that there is 15 seconds of difference between each pair of coordinates.
# 
#     TRIP_START: (string): It contains the first pair of coordinates of POLYLINE.
# 
#     AVERAGE_SPEED: (float): It contains the average speed calculated from trip distance and trip time.
# 
#     TOP_SPEED: (float): It contains the top speed calculated from distances and time of all pairs of coordinates.
# 
#     YEAR: (integer): It contains the year of the timestamp.
# 
#     MONTH: (integer): It contains the month of the timestamp.
# 
#     DAY: (integer): It contains the day of the timestamp.
# 
#     HOUR: (integer): It contains the hour of the timestamp.
# 
#     WEEKDAY: (integer): It contains the day of the week of the timestamp.

# # Import libraries

# In[ ]:


import pandas as pd
import numpy as np
import multiprocessing as mp
import geopy.distance
import csv
from datetime import datetime, timezone
from tqdm import tqdm

tqdm.pandas()
# %load_ext nb_black


# # Load data

# In[ ]:


taxi = pd.read_csv(
    "../input/train.csv",
    sep=",",
    low_memory=True,
#             skiprows=lambda i: i % 10 != 0,  # Use only 1 of each n
)


# # Extract date components from timestamp

# In[ ]:


taxi["YEAR"] = taxi.TIMESTAMP.apply(lambda x: datetime.fromtimestamp(x).year)
taxi["MONTH"] = taxi.TIMESTAMP.apply(lambda x: datetime.fromtimestamp(x).month)
taxi["DAY"] = taxi.TIMESTAMP.apply(lambda x: datetime.fromtimestamp(x).day)
taxi["HOUR"] = taxi.TIMESTAMP.apply(lambda x: datetime.fromtimestamp(x).hour)
taxi["WEEKDAY"] = taxi.TIMESTAMP.apply(
    lambda x: datetime.fromtimestamp(x).weekday()
)


# # Cleaning

# In[ ]:


taxi.TRIP_ID.count()


# ## Deleting duplicates

# In[ ]:


taxi = taxi.drop_duplicates()


# In[ ]:


taxi.TRIP_ID.count()


# ## Deleting missing data

# In[ ]:


taxi = taxi[taxi.MISSING_DATA == False]


# In[ ]:


taxi.TRIP_ID.count()


# ## Deleting void polylines

# In[ ]:


taxi = taxi[taxi.POLYLINE != "[]"]


# In[ ]:


taxi.TRIP_ID.count()


# ## Reseting index

# In[ ]:


taxi.reset_index(drop=True, inplace=True)


# # Extract trip start
# 
# Extract the first pair of coordinates as [longitude, latitude]

# In[ ]:


def map_trip_start(df_in):
    df_out = pd.DataFrame()
    df_out["TRIP_START"] = df_in["POLYLINE"].progress_map(lambda x: repr(eval(x)[0]))
    return df_out


# In[ ]:


p = mp.Pool(processes=mp.cpu_count())
pool_results = p.map(map_trip_start, np.array_split(taxi, mp.cpu_count()))
p.close()
p.join()

# merging parts processed by different processes
new_columns = pd.concat(pool_results, axis=0, ignore_index=True)

# merging newly calculated columns to taxi
taxi = pd.concat([taxi, new_columns], axis=1)


# # Calculate trip distance, trip time, average speed and top speed per trip
# 
# The trip distance is the sum of all the distances between each two correlative pairs of coordinates in kilometers.
# 
# Each pair of coordinates are separate 15 seconds in time. Then, the trip time in minutes will be: (number of pair of coordinates -1) x 15 seconds x 1 minute / 60 seconds.
# 
# The average speed is simply the trip distance divided by the trip time in hours. The result is in km/h.
# 
# The top speed is the greatest speed of all the sections of the trip. The speed of one section is calculated as the distance of the section divided by 15 seconds and converted to km/h.

# In[ ]:


def process_polyline(p):
    p = eval(p)
    if len(p) > 1:
        trip_distance = 0
        top_speed = 0
        for i in range(len(p) - 1):
            distance = geopy.distance.distance(p[i], p[i + 1]).km
            trip_distance += distance
            speed = distance / 15 * 3600
            if speed > top_speed:
                top_speed = speed
        trip_time = (len(p) - 1) * 15 / 60
        avg_speed = trip_distance / trip_time * 60
        return trip_distance, trip_time, avg_speed, top_speed
    else:
        return np.NaN, np.NaN, np.NaN, np.NaN


# Testing on one specific trip:

# In[ ]:


print(
    "Trip distance: {:>5.1f} km\n"
    "Trip time:     {:>5.1f} min\n"
    "Average speed: {:>5.1f} km/h\n"
    "Top speed:     {:>5.1f} km/h".format(*process_polyline(taxi.POLYLINE[0]))
)


# In[ ]:


def map_polyline(df_in):
    df_out = pd.DataFrame()
    df_out["TRIP_DISTANCE"], df_out["TRIP_TIME"], df_out[
        "AVERAGE_SPEED"
    ], df_out["TOP_SPEED"] = zip(
        *df_in["POLYLINE"].progress_map(process_polyline)
    )
    return df_out


# In[ ]:


p = mp.Pool(processes=mp.cpu_count())
pool_results = p.map(map_polyline, np.array_split(taxi, mp.cpu_count()))
p.close()
p.join()

# merging parts processed by different processes
new_columns = pd.concat(pool_results, axis=0, ignore_index=True)

# merging newly calculated columns to taxi
taxi = pd.concat([taxi, new_columns], axis=1)


# Save results as processing polylines are very time consuming

# In[ ]:


taxi.to_csv("train_extended.csv.zip", index=None, compression="zip")

