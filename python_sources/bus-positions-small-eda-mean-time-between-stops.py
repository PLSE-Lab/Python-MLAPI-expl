#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, text
from geopy.distance import distance
import math
import time
from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True) 


# ### Chose a route

# In[ ]:


route = '1012-10'


# **Showing all the stop IDs for this route.**

# In[ ]:


bus_stops = pd.read_csv('../input/so-paulo-bus-system/stops.csv')
stop_times = pd.read_csv('../input/so-paulo-bus-system/stop_times.csv')
trips_id = pd.read_csv('../input/so-paulo-bus-system/trips.csv')

stops = pd.merge(stop_times, bus_stops, on='stop_id')
stops = pd.merge(stops, trips_id, on='trip_id')

stops = stops[stops['route_id'].str.contains(route)]

stop_names = stops.drop_duplicates(['stop_id', 'stop_name'])[['stop_id', 'stop_sequence', 'stop_name']]
stop_names.head()


# ### Chose the stop IDs. They must be sequential.

# In[ ]:


stops_chosen = (301729, 301764)


# In[ ]:


stops = stops[(stops['stop_id'] == stops_chosen[0]) | (stops['stop_id'] == stops_chosen[1])] 
stops.drop_duplicates('stop_id', inplace=True)


# ## Reading bus positions csv.
# 
# **We are reading chunks of the dataset because of its size (3 GB). The chunks are going to be filtered by the chosen route and stored in a data frame.**
# 

# In[ ]:


reader = pd.read_csv('../input/so-paulo-bus-system/bus_position.csv', chunksize=10 ** 6)
bus_position = pd.DataFrame()
for i, chunk in enumerate(reader):
    print(i)
    chunk = chunk[chunk['c'].str.contains(route)]
    bus_position = pd.concat([bus_position, chunk])


# In[ ]:


bus_position.shape


# In[ ]:


bus_position.info()


# In[ ]:


bus_position.head()


# ## Converting strings to time

# In[ ]:


bus_position['hr'] = pd.to_datetime(bus_position['hr']).dt.time


# In[ ]:


bus_position['ta'] = pd.to_datetime(bus_position['ta']).dt.tz_convert('America/Sao_Paulo')


# In[ ]:


print(f'First observation was on {bus_position["ta"].min()}')
print(f'Last observation was on {bus_position["ta"].max()}')


# ## Shifting Data Frame
# **If we want to calculate time intervals, distances, and speeds, we must shift the data and concatenate it to the original data frame.**

# In[ ]:


bus_position.sort_values(['p', 'ta'], ignore_index=True, inplace=True)


# In[ ]:


bp_copy = bus_position.copy()


# In[ ]:


bp_copy.head()


# In[ ]:


grouped_cl = bp_copy.groupby('p')


# In[ ]:


shifted_values = grouped_cl.shift(1)[['ta', 'py','px']]


# In[ ]:


shifted_values.columns = ['previous_ta', 'previous_py', 'previous_px']


# In[ ]:


bp_copy = pd.concat([bp_copy, shifted_values], axis=1)


# ## Calculating distance in KM
# **We were not sure how we were going to use this data. Therefore, we calculated the distance in order to try to find some insights**

# In[ ]:


bp_copy['latlon'] = pd.Series(list(bp_copy[['py', 'px']].itertuples(index=False, name=None)))


# In[ ]:


bp_copy['previous_latlon'] = pd.Series(list(bp_copy[['previous_py', 'previous_px']].itertuples(index=False, name=None)))


# In[ ]:


def calc_distance(coord1, coord2):
  if type(coord1) is tuple and type(coord2) is tuple: 
    if np.isnan(coord1[0]) or np.isnan(coord1[1]):
      return np.nan

    elif np.isnan(coord2[0]) or np.isnan(coord2[1]):
      return np.nan

    else:
      return distance(coord1, coord2).km
    
  else:

    return np.nan


# In[ ]:


bp_copy['distance'] = bp_copy.apply(lambda x: calc_distance(x['latlon'], x['previous_latlon']),axis=1)


# In[ ]:


bp_copy.head()


# ## Distance analysis

# In[ ]:


plt.figure(figsize=[12,6])
sns.distplot(bp_copy['distance'])


# **There is a lot of zeroes in the distance. Let's analyse only buses in movement.**

# In[ ]:


plt.figure(figsize=[12,6])
sns.distplot(bp_copy[bp_copy['distance'] != 0]['distance'])


# **A little bit more of information of the moving buses.**

# In[ ]:


bp_copy[bp_copy['distance'] != 0]['distance'].describe()


# In[ ]:


plt.figure(figsize=[12,6])
sns.distplot(np.log(bp_copy[bp_copy['distance'] != 0]['distance'] + 1))


# ## Calculating timedelta

# In[ ]:


bp_copy['timedelta'] = bp_copy.apply(
            lambda x: (x['ta'] - x['previous_ta']) if x['previous_ta'] else None, axis=1)


# In[ ]:


def deltatime_to_float(delta):
    if type(delta) is pd._libs.tslibs.timedeltas.Timedelta:
        return delta.to_timedelta64().astype('timedelta64[s]').item().total_seconds() / 3600

    else:
        return None


# In[ ]:


bp_copy['timedelta'] = bp_copy['timedelta'].apply(deltatime_to_float)


# In[ ]:


bp_copy.head()


# # Calculating Historical Average Time Between Stops

# **All the stops data frame merges and filters have been applied in the first cells**

# ## From now on we are going to work with just one bus route for efficiecy purposes

# In[ ]:


i_list = []
index_list = []
id_list = []
for index, row in stops[stops['route_id'].str.contains('1012-10')].iterrows():
  i_list = []
  i_list.append(bp_copy[(bp_copy['c'].str.contains('1012-10')) & 
                    (((bp_copy['previous_py'] <= row['stop_lat']) & (row['stop_lat'] <= bp_copy['py']) & (bp_copy['previous_px'] <= row['stop_lon']) & (row['stop_lon'] <= bp_copy['px'])) | 
                    ((bp_copy['py'] <= row['stop_lat']) & (row['stop_lat'] <= bp_copy['previous_py']) & (bp_copy['px'] <= row['stop_lon']) & (row['stop_lon'] <= bp_copy['previous_px'])) | 
                    ((bp_copy['py'] <= row['stop_lat']) & (row['stop_lat'] <= bp_copy['previous_py']) & (bp_copy['previous_px'] <= row['stop_lon']) & (row['stop_lon'] <= bp_copy['px'])) | 
                    ((bp_copy['previous_py'] <= row['stop_lat']) & (row['stop_lat'] <= bp_copy['py']) & (bp_copy['px'] <= row['stop_lon']) & (row['stop_lon'] <= bp_copy['previous_px'])))]['id'].values)
  for i_value in i_list:
    for value in i_value:
      index_list.append(index)
      id_list.append(value)
association_df = pd.DataFrame({'index': index_list, 'id': id_list})
association_df


# In[ ]:


association_df['ta'] = association_df.apply(lambda x: bp_copy[bp_copy['id'] == x['id']]['ta'].values[0], axis = 1)
association_df['previous_ta'] = association_df.apply(lambda x: bp_copy[bp_copy['id'] == x['id']]['previous_ta'].values[0], axis = 1)
association_df['stop_sequence'] = association_df.apply(lambda x: stops[stops.index == x['index']]['stop_sequence'].values[0], axis=1)
association_df 


# In[ ]:


def calculate_real_time(stop_lat, previous_py, stop_lon, previous_px, py, px, time):
  value = math.sqrt((stop_lat - previous_py)**2 + (stop_lon - previous_px)**2)
  value = value/math.sqrt((py - previous_py)**2 + (px - previous_px)**2)
  return time*value


# In[ ]:


association_df['time'] = association_df.apply(lambda x: calculate_real_time(stops.iloc[x['index']]['stop_lat'], bp_copy[bp_copy['id'] == x['id']]['previous_py'], stops.iloc[x['index']]['stop_lon'],
                                                                            bp_copy[bp_copy['id'] == x['id']]['previous_px'], bp_copy[bp_copy['id'] == x['id']]['py'], bp_copy[bp_copy['id'] == x['id']]['px'],
                                                                            ((time.mktime(x['ta'].timetuple())/60) - (time.mktime(x['previous_ta'].timetuple())/60))), axis = 1)
association_df


# In[ ]:


def validate_time(df, stop_time, ta, index):
    return df[(df['ta'] == ta) & (df['stop_sequence'] == index - 1)]['time'].size != 0 and stop_time - df[(df['ta'] == ta) &
           (df['stop_sequence'] == index - 1)]['time'].values[0] > 0


def get_time_between_stops(index, ta, previous_ta, bus_stop_time, association_df):
    if index > 1:
        if validate_time(association_df, bus_stop_time, ta, index):
            return bus_stop_time - association_df[(association_df['ta'] == ta) & (association_df['stop_sequence'] == index - 1)
                                                  ]['time'].values[0]

        if association_df[(association_df['ta'] == previous_ta) & (association_df['stop_sequence'] == index - 1)]['time'].size != 0:

            time_between_stops = bus_stop_time - association_df[(association_df['ta'] == previous_ta) &
                                                                (association_df['stop_sequence'] == index - 1)]['time'].values[0]

            time_between_stops = time_between_stops + (association_df[(association_df['ta'] == previous_ta) &
                                                       (association_df['stop_sequence'] == index - 1)]['ta'].values[0] -
                                                       np.datetime64('1970-01-01T00:00:00Z'))/np.timedelta64(1, 's')/60

            return time_between_stops - (association_df[(association_df['ta'] == previous_ta) &
                                         (association_df['stop_sequence'] == index - 1)]['previous_ta'].values[0] -
                                         np.datetime64('1970-01-01T00:00:00Z'))/np.timedelta64(1, 's')/60
    return np.nan


# In[ ]:


association_df.head()


# In[ ]:


association_df['time_between_stops'] = association_df.apply(
       lambda x: get_time_between_stops(x['stop_sequence'], x['ta'], x['previous_ta'], x['time'], association_df),
                                                                                                                axis=1)


# In[ ]:


association_df[association_df['time_between_stops'].notna()]


# In[ ]:


association_df.groupby(['index'])['time_between_stops'].mean()


# In[ ]:


# association_df[association_df['time_between_stops'].notna()]['time_between_stops'].mean()


# In[ ]:


# association_df[association_df['time_between_stops'].notna()].groupby(['index', 'weekday'])['time_between_stops'].mean()


# In[ ]:


# association_df['weekday'] = association_df['previous_ta'].dt.weekday
# association_df['hour'] = association_df['previous_ta'].dt.


# In[ ]:


association_df['ta'].dt.day.unique()


# In[ ]:


association_df.set_index(['ta'], inplace=True)
association_df = association_df[association_df['time_between_stops'] < 10]

agg = association_df.groupby([association_df.index.hour, association_df.index.day_name()]).mean()

days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
agg = agg.reindex(days, level=1)


# In[ ]:


association_df.sort_index()


# In[ ]:


agg[agg['weekday'] == 'Friday']


# In[ ]:


agg.index.names = ['hour', 'weekday']
agg = agg.reset_index()

agg['hour'] = agg['hour'].apply(lambda x: str(x).zfill(2) + ':' + '00')


# In[ ]:


filtered = agg[agg['hour'] == '18:00']


# In[ ]:


fig = px.bar(filtered, x='weekday', y='time_between_stops', title=f'Time it takes between stops')


# In[ ]:


fig.show()


# In[ ]:




