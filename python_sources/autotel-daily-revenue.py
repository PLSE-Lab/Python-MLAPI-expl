#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import tqdm
import datetime
import matplotlib.pyplot as plt
import os
import geopy.distance
import folium
import seaborn as sns
from sklearn.cluster import KMeans

print(os.listdir("../input"))


# ## Calculating AutoTel's Daily Revenue
# This notenook is a fork of @drgilermo's notebook that extracted from the AutoTel dataset the actual trips by examining when and where parked cars appear on the map.
# The Addition is a simple group by that adds the total duration time and multiples it by a fixed price of 1.3 ILS per minute which is the assumed average price per minute ride.

# # Read The Data

# In[ ]:


df = pd.read_csv(r'../input/autotel-shared-car-locations//sample_table.csv')


# # Some Transformations

# In[ ]:


df['carsList'] = df.carsList.apply(lambda x: x[1:-1]) # remove square brackets
df['carsList'] = df.carsList.apply(lambda x: x.split(',')) # convert string to list
df['carsList'] = df.carsList.apply(lambda x: [] if x == [''] else x) # denote empty lists
df['carsList'] = df.carsList.apply(lambda x: [int(i) for i in x]) # convert list items to int
df['total_cars'] = df.carsList.apply(len) 
df = df[df.total_cars > 0]


# In[ ]:


# Parse list of cars into different rows 
def explode(df, lst_cols, fill_value=''):
    # make sure `lst_cols` is a list
    if lst_cols and not isinstance(lst_cols, list):
        lst_cols = [lst_cols]
    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)

    # calculate lengths of lists
    lens = df[lst_cols[0]].str.len()

    if (lens > 0).all():
        # ALL lists in cells aren't empty
        return pd.DataFrame({
            col:np.repeat(df[col].values, lens)
            for col in idx_cols
        }).assign(**{col:np.concatenate(df[col].values) for col in lst_cols}) \
          .loc[:, df.columns]
    else:
        # at least one list in cells is empty
        return pd.DataFrame({
            col:np.repeat(df[col].values, lens)
            for col in idx_cols
        }).assign(**{col:np.concatenate(df[col].values) for col in lst_cols}) \
          .append(df.loc[lens==0, idx_cols]).fillna(fill_value) \
          .loc[:, df.columns]
    
new_df = explode(df, ['carsList'], fill_value='')


# In[ ]:


# Pivot the table to a new structures, where the indices are unique timestamps, the columns are cars and the values are the coordinates of the cars
pivot_df = new_df.pivot(index='timestamp',columns='carsList', values=['latitude', 'longitude'])


# In[ ]:


pivot_df.head()


# Some pandas transformaton of the data. If anyone has a more elegant solution I would be happy to incorporate it into my code

# In[ ]:


def get_car_trips(pivot_df, car_num):
    # First, take the relevant columns for the car in question
    car = pivot_df[[('latitude', car_num), ('longitude', car_num)]]
    car = car[pd.isnull(car[('latitude', car_num)]) == False]
    
    # Find the previous location
    car.loc[:, 'prev_lat'] = car.shift()[('latitude', car_num)]
    car.loc[:, 'prev_lon'] = car.shift()[('longitude', car_num)]
    
    # If the location has not changed, there is no trip going on
    car.loc[:, 'trip'] = car[('latitude', car_num)] == car.prev_lat
    car.loc[:, 'trip'] = car.trip.apply(lambda x: 0 if x else 1)
    car.loc[:, 'trip'] = car.trip.cumsum()
    car.reset_index(inplace=True)
    
    # Merge the data frame with itself shifted by one
    f = {'timestamp': ['min', 'max'], ('latitude', car_num): 'first', ('longitude', car_num): 'first'}
    trip_df = car.groupby('trip').agg(f)
    prev_df = car.groupby('trip').agg(f).shift()

    trip_df = pd.merge(trip_df, prev_df, left_index=True, right_index=True)
    
    trip_df.columns = trip_df.columns.get_level_values(0)
    trip_df.columns = ['end', 'start_next', 'end_lat', 'end_long', 'end_prev', 'start', 'start_lat', 'start_long']
    trip_df['car'] = car_num
    return trip_df


# In[ ]:


trips = pd.DataFrame()

for car in tqdm.tqdm(np.array(pivot_df.columns.get_level_values(1))):
    trips = trips.append(get_car_trips(pivot_df, car))
    
print(len(trips))


# # Trips data frame
# Now we have a new data frame, where each row is a trip, with a starting and an ending point, as well as the relevant timestamps and car id. Let's add the interesting columns to each trip

# In[ ]:


def trip_distance(lat1, lat2, lon1, lon2):
    try:
        coords_1 = (lat1, lon1)
        coords_2 = (lat2, lon2)
        return geopy.distance.vincenty(coords_1, coords_2).km
    except ValueError:
        return -1


# In[ ]:


trips['trip_len'] = trips.apply(lambda x: trip_distance(x.start_lat, x.end_lat, x.start_long, x.end_long), axis=1)

trips.reset_index(inplace=True)


# In[ ]:


def transform_time(x):
    try:
        return datetime.datetime.strptime(x[:19], '%Y-%m-%d %H:%M:%S')
    except TypeError:
        return -1
trips['end'] = trips.end.apply(transform_time)
trips['start'] = trips.start.apply(transform_time)


# In[ ]:


trips = trips[trips.trip_len > -1]
trips['trip_duration'] = trips.apply(lambda x: (x.end - x.start).seconds/60, axis=1)
trips = trips[trips.trip_duration > 3]


# In[ ]:


import seaborn as sns
plt.figure(figsize=(8, 6))
plt.style.use('fivethirtyeight')
sns.distplot(trips.trip_duration, bins=np.linspace(0, 120, 60), kde=False)
plt.xlabel('Duration [mins]')


# ## Analizing total revene
# Here we simply group the table by day to sum the total rides and ride durations and multiplie by the assumed average cost per minute of ride

# In[ ]:


trips.head()


# In[ ]:


# lets count the number of rides per day
df_rides = trips.copy()
df_rides['start'] = df_rides['start'].apply(pd.Timestamp)
df_rides = df_rides.set_index('start')

df_rides = df_rides[df_rides['trip_duration'] < 90] # very long trips are maintanance time

df_daily_rids = df_rides.groupby(pd.Grouper(freq='1D')).agg({'trip':'count', 'trip_duration':'sum'})
df_daily_rids.index = df_daily_rids.index.map(lambda t: t.strftime('%Y-%m-%d'))
df_daily_rids.head()


# In[ ]:


df_daily_rids['trip'].plot(kind='bar', figsize=(20,6))
plt.title("Daily recorded trips")
plt.ylabel('Total trips')
plt.show()


# In[ ]:


df_daily_rids['trip_duration'].plot(kind='bar', figsize=(20,6))
plt.title("Total daily ride time")
plt.ylabel('Ride time (minutes)')
plt.show()


# In[ ]:


# We assume all rides cost 1.3 ILS per minute
df_daily_rids['daily_revenue'] = (df_daily_rids['trip_duration'] * 1.3) / 1000

df_daily_rids['daily_revenue'].plot(kind='bar', figsize=(20,6))
plt.title("Autotel Daily Revenue")
plt.ylabel('1K ILS')
plt.show()


# In[ ]:


df_daily_rids[['daily_revenue']].plot(kind='hist', bins=range(40, 190, 10), alpha=0.6, figsize=(14,5))
plt.show()


# In[ ]:


# The average daily revenue is estimated to be around 100K ILS :)

