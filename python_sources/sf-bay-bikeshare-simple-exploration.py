#!/usr/bin/env python
# coding: utf-8

# # Init

# In[ ]:


import os
import sys


# In[ ]:


# if you don't want the data to be downladed from kaggle assign the correct data path here
data_dir = None


# In[ ]:


if data_dir is not None:
    pass
elif os.path.isdir('/kaggle'):
    data_dir = '../input'
    assert(os.path.isdir(data_dir))
else:
    script_dir = os.getcwd()
    data_dir = os.path.join(script_dir, 'sf-bay-area-bike-share')

if os.path.isdir(data_dir):
    print(f"Using existing data directory {data_dir}")
else:
    print(f"Data directory {data_dir} not found. Need to download from kaggle")
    if 'google.colab' in sys.modules and not os.path.isfile(os.path.expanduser('~/.kaggle/kaggle.json')):
        import google.colab
        print("Upload your kaggle.json")
        google.colab.files.upload()
        get_ipython().system('mkdir -p ~/.kaggle')
        get_ipython().system('mv kaggle.json ~/.kaggle/')
        get_ipython().system('chmod 600 ~/.kaggle/kaggle.json')
    try:
        import kaggle
    except:
        get_ipython().system('pip install kaggle')
    get_ipython().system('kaggle datasets download -d benhamner/sf-bay-area-bike-share')
    get_ipython().system('mkdir -p "$data_dir"')
    get_ipython().system('unzip -d "$data_dir" sf-bay-area-bike-share.zip')

if not os.path.isdir(data_dir):
    raise ValueError(f"Data directory {data_dir} still not found. Ensure correct kaggle setup.")


# In[ ]:


station_csv = os.path.join(data_dir, 'station.csv')
status_csv = os.path.join(data_dir, 'status.csv')
trip_csv = os.path.join(data_dir, 'trip.csv')
weather_csv = os.path.join(data_dir, 'weather.csv')


# In[ ]:


get_ipython().system('pip install -q numpy pandas matplotlib folium')


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import folium
import folium.plugins


# # Load data

# In[ ]:


station = pd.read_csv(station_csv, index_col='id', parse_dates=['installation_date'])
station.info()


# In[ ]:


trip = pd.read_csv(trip_csv,
                   index_col='id',
                   usecols=['id', 'duration', 'start_date', 'start_station_id', 'end_date', 'end_station_id', 'bike_id', 'subscription_type'],
                   parse_dates=['start_date', 'end_date'],
                   infer_datetime_format=True,
                   dtype={'duration': np.uint32, 'start_station_id': np.uint8, 'end_station_id': np.uint8, 'bike_id': np.uint16},
                   true_values=['Subscriber'],
                   false_values=['Customer'],
                   )
trip.info()


# In[ ]:


# Uh oh
# status_size = 1000000
# status = pd.read_csv(status_csv,
#                      index_col='time',
#                      parse_dates=['time'],
#                      infer_datetime_format=True,
#                      dtype={'station_id': np.uint8, 'bikes_available': np.uint8, 'docks_available': np.uint8},
#                      low_memory=True,
#                      memory_map=True,
#                      nrows=status_size,
#                      )
# status.info()
# print(status.describe())


# # Stations

# In[ ]:


station.head()


# ## Station size
# in terms of number of docks

# In[ ]:


station.dock_count.describe()


# In[ ]:


station.dock_count.hist(bins=18)


# ## Station installation time

# In[ ]:


(station.installation_date.min(), 
 station.installation_date.max(), 
 station.installation_date.max() - station.installation_date.min())


# In[ ]:


station.installation_date.hist(bins=24, xrot=90);


# Almost all stations are installed on August-September of 2013.

# ## Station installation place

# In[ ]:


station.groupby('city').size().plot.bar(title="Number of stations per city");


# Most of the stations are in San Francisco. Maybe these are small stations, so the total capacity per city is not as high?

# In[ ]:


station.groupby('city').agg({'dock_count': 'sum'}).plot.bar(title="Numbe of docks per city");


# Nope, still the same picture. Speaking of which, let's visualize stations on the map (using station size for density).

# In[ ]:


station_hm = station.loc[:, ['lat', 'long']].copy()
station_hm.loc[:, 'dock_count_norm'] = (station.dock_count - station.dock_count.min()) / (station.dock_count.max() - station.dock_count.min())

station_map = folium.Map(location=[station.lat.mean(), station.long.mean()], zoom_start=10)
folium.plugins.HeatMap(station_hm.values.tolist()).add_to(station_map)
#station_map.save(os.path.join(data_dir, 'station_map.html'))
station_map


# # Trips

# In[ ]:


trip.head()


# ## Trip duration

# In[ ]:


trip.duration.describe()


# Trip duration has a very long tail. Let's see its quantiles and plot a distribution.

# In[ ]:


print("Trip duration quantiles")
trip.duration.quantile([0.5, 0.9, 0.99, 0.999, 0.9999, 1]).apply(
    lambda x: pd.Series({"minutes": x/60, "hours": x/3600, "days": x/86400}))


# In[ ]:


fig, ax = plt.subplots(ncols=2, figsize=(15, 5))
for (quantile, axis) in zip([0.95, 0.99], ax):
    f=(trip[trip.duration < trip.duration.quantile(quantile)].duration / 60).hist(log=True, bins=100, ax=axis)
    f.set(title=f"Trips by duration ({quantile} quantile)", xlabel="duration (minutes)", ylabel="number of trips")


# ## Longer rents
# Some bike rents are too long to be a single trip. Let's take a look on rents longer than 12 hours.

# In[ ]:


long_rent_threshold_days = 0.5
long_rent=trip[trip.duration > 86400*long_rent_threshold_days]
(long_rent.duration/3600).describe()


# In[ ]:


hist, bins = np.histogram(long_rent.duration/86400, np.geomspace(long_rent_threshold_days, 200, num=51))
fig,ax = plt.subplots()
ax.bar(range(len(hist)),hist,width=1) 
#ax.set_yscale("log")
ax.set_xticks(range(0, len(hist)+1, 5));
ax.set_xticklabels(np.round(bins[::5]).astype(np.int), rotation=90);
ax.set_title("Long bike rents (at least 12 hours)")
ax.set_xlabel("rent duration (days)")
ax.set_ylabel("number of rents");


# Among the longer rented bikes 75% is returned within 24 hours and almost all within a week.

# ## Inter-city trips
# Are there any people who ride bike between cities?

# In[ ]:


start_station=station.rename(columns=lambda s: f'ss_{s}')
end_station=station.rename(columns=lambda s: f'es_{s}')
trip_station = trip.join(start_station, on='start_station_id').join(end_station, on='end_station_id')

intercity_trips=trip_station[trip_station.ss_city != trip_station.es_city]
print("Number of intercity rents:", len(intercity_trips.index))
intercity_trips = intercity_trips[intercity_trips.duration < 12*3600]
print("Number of intercity trips (rents shorter than 12 hours):", len(intercity_trips.index))


# In[ ]:


(intercity_trips.duration / 3600).hist(log=True, bins=100).set(
    title=f"Intercity trips by duration", xlabel="duration (hours)", ylabel="number of trips");


# In[ ]:


(intercity_trips.duration/60).describe()


# Median intercity trip is below half an hour. Probably most of them are between neighboring cities.

# In[ ]:


intercity_trips.groupby(['ss_city', 'es_city']).size().unstack().fillna(0)


# Indeed, most intercity trips are between Palo Alto and Mountain View with trips between Palo Alto and Redwood City is a distant second. We can visualize the stations on a map and show number of trips originating at the station.

# In[ ]:


intercity_hm=intercity_trips.groupby(['start_station_id', 'ss_lat', 'ss_long']).size().reset_index().rename(columns={0: 'count'})
intercity_hm = intercity_hm[intercity_hm['count'] > 5]
intercity_hm.loc[:, 'count_norm'] = intercity_hm['count'] / intercity_hm['count'].max()

station_map2 = folium.Map(location=[intercity_hm.ss_lat.mean(), intercity_hm.ss_long.mean()], zoom_start=12)
folium.plugins.HeatMap(intercity_hm[['ss_lat', 'ss_long', 'count_norm']].values.tolist()).add_to(station_map2)
for _, row in intercity_hm.iterrows():
    number_icon = folium.plugins.BeautifyIcon(
        text_color='#000', border_color='transparent',
        background_color='transparent', number=row['count'],
        inner_icon_style='font-size:12px;padding-top:-5px;')
    folium.Marker(location=[row['ss_lat'], row['ss_long']],
                  popup=folium.Popup('station #{}, trips: {}'.format(int(row['start_station_id']), row['count'])),
                  icon=number_icon).add_to(station_map2)
station_map2


# ## Trips by city and efficiency

# In[ ]:


trip_station.groupby('ss_city').size().plot.bar(logy=True).set(title="Numbe of trips per city", xlabel='');


# No surprises here, but how efficient is bike stations allocation? Let's normalize number of trips by number of bike docks installed in the city.

# In[ ]:


(trip_station.groupby('ss_city').size() / 
 station.groupby('city').agg({'dock_count': 'sum'}).dock_count).plot.bar().set(
    title="Efficiency: number of trips per dock installed", xlabel=''#, ylabel='as'
);


# Not only San Francisco has most bikes deployed, but each bike makes much more trips on average than in other cities.

# ## Pedalling together
# Are there people who travel together by bike? Let's find how many people start and finish bike trip together. (We do not count here people who start/finish not on the same minute, though)

# In[ ]:


group_trip=trip.groupby(['start_date', 'start_station_id', 'end_date', 'end_station_id']).size()
group_hist=np.histogram(group_trip.values, bins=np.arange(1,group_trip.max()+2))


# In[ ]:


fig, ax = plt.subplots()
plt.bar(x=group_hist[1][:-1], height=group_hist[0]) #, bins=group_hist[1])
ax.set_yscale('log')
ax.set_title('trips by group size')
ax.set_xlabel('group size')
ax.set_ylabel('number of trips')
for (y, x) in list(zip(*group_hist)):
    ax.text(x, y+1, str(y), horizontalalignment='center', verticalalignment='bottom', fontsize=12)


# Most people ride alone. There are, however, 16K trips of pairs and even a trip of seven people!

# In[ ]:


df1=group_trip[group_trip<=4].reset_index().rename(columns={0: "group_size"})
df2=df1.groupby(['group_size', 'start_station_id']).agg({'group_size': 'count'}).rename(columns={'group_size': 'group_count'})
df_count=df1.groupby('group_size').agg({'group_size': 'count'}).rename(columns={'group_size': 'total_group_count'})
df3=df2.reset_index().join(df_count, on='group_size')
df3['station_group_ratio'] = df3['group_count'] / df3['total_group_count']
df4=df3.drop(columns=['group_count', 'total_group_count']).set_index(['start_station_id', 'group_size']).unstack().fillna(0)
df4.columns=df4.columns.droplevel()


# How stations are used by inidividual riders?

# In[ ]:


df4.loc[:, 1].plot.bar(figsize=(20, 10)).set(
    title='Station popularity among single riders', xlabel='station id', ylabel='rider ratio');


# San Francisco stations used more, as expected. How is station popularity distributed among groups of 2-4 people?

# In[ ]:


df4.loc[:, 2:4].plot.bar(figsize=(20, 10), stacked=True).set(
    title='Station popularity among group riders', xlabel='station id', ylabel='rider ratio');


# While San Francisco domitnates here as well, station popularity amoung groups differ from that of single riders.

# ---
