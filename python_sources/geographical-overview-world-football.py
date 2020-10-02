#!/usr/bin/env python
# coding: utf-8

# Hello Everyone,
# 
# This is one of my first ever kernels on this website. My aim is to be able to generate a world map where all world football matches have been played.
# 
# Moreover, I will generate a few additional features for anyone to use to explore further.
# 
# I am using two additional datasets from Kaggle, to borrow the longitude and latitude coordinates of various cities where these matches have been played.

# In[ ]:


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

results = pd.read_csv('../input/international-football-results-from-1872-to-2017/results.csv')
results.head(10)


# I have extracted the day, month, year values from the date as it is more usefull. Additionally, I have added a new column to check who won each game and in case there was a draw.
# 
# Although I will not be using these new features in this notebook, anyone is welcome to borrow the code to use for further analysis.

# In[ ]:


results['year'] = pd.to_numeric([date.split('-')[0] for date in results['date']])
results['month'] = pd.to_numeric([date.split('-')[1] for date in results['date']])
results['day'] = pd.to_numeric([date.split('-')[2] for date in results['date']])

def find_winners(df):
    winners = []
    for i, row in df.iterrows():
        if row['home_ft'] > row['away_ft']:
            winners.append(row['home_team'])
        elif row['home_ft'] < row['away_ft']:
            winners.append(row['away_team'])
        else:
            winners.append('Draw')
    return winners
results['winner'] = find_winners(results)
results.head()


# I noticed that the mention for matches played at a neutral venue was missing from the dataset. I have added the neutral venue matches indicator as a new column.

# In[ ]:


neutral = []
for index, row in results.iterrows():
    neutral.append((row['country'] not in  row['home_team']) and (row['home_team'] != 'USA'))
    
results['neutral'] = neutral
results['neutral'] = results['neutral'].astype(int)

results.head()


# I have borrowed data from the World Cities dataset to finc latitude and longitude values.

# In[ ]:


cities = pd.read_csv('../input/world-cities-database/worldcitiespop.csv', encoding='latin-1', dtype={
    'Country': str,
    'City': str,
    'AccentCity': str,
    'Region': str
})
cities = cities.dropna()
cities = cities[['Country', 'AccentCity', 'Latitude', 'Longitude']]
cities.head()


# In[ ]:


countries = pd.read_csv('../input/world-countries-and-continents-details/countries and continents.csv')
countries = countries.rename(columns = {'official_name_en': 'Name', 'ISO3166-1-Alpha-2': 'Code'})
countries = countries[['Name', 'Code']]
countries['Code'] = countries['Code'].str.lower()


# In[ ]:


coords = results[['city', 'country']]
df = coords.merge(cities, how='inner', left_on='city', right_on='AccentCity')
new_df = df.merge(countries, how='left', left_on='Country', right_on='Code')
city_coords = new_df[['city', 'Latitude', 'Longitude']]
city_coords = city_coords.drop_duplicates('city')
city_coords = city_coords.rename(columns={'Latitude':'latitude', 'Longitude': 'longitude'})


# Finally, I have merged the geographical coordinates data to the original dataset and is available in the combined dataframe named 'new_results'.

# In[ ]:


new_results = results.merge(city_coords, how='left', on='city')
new_results.head()


# A World Map to see where football has been played most:

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from mpl_toolkits.basemap import Basemap

lat_min = min(new_results['latitude'])
lat_max = max(new_results['latitude'])
lon_min = min(new_results['longitude'])
lon_max = max(new_results['longitude'])

m = Basemap(
    projection='merc', 
    llcrnrlat=lat_min, 
    urcrnrlat=lat_max, 
    llcrnrlon=lon_min, 
    urcrnrlon=lon_max,
    resolution='l'
)


longitudes = new_results['longitude'].tolist()
latitudes = new_results['latitude'].tolist()
fig = plt.figure(figsize=(30,30))
ax=fig.add_subplot(1,1,1)
ax = m.drawcountries()
ax = m.drawcoastlines()
ax = m.fillcontinents(color='#eadf93')
ax = m.drawmapboundary()
ax = m.scatter(longitudes, latitudes, s=25, zorder=2, latlon=True, color='#0e3bef')


# I have tailored the map a little to check where FIFA World Cup matches have been played over the world:

# In[ ]:


world_cup = new_results[new_results['tournament']=='FIFA World Cup']

m = Basemap(
    projection='merc', 
    llcrnrlat=lat_min, 
    urcrnrlat=lat_max, 
    llcrnrlon=lon_min, 
    urcrnrlon=lon_max,
    resolution='i'
)


longitudes = world_cup['longitude'].tolist()
latitudes = world_cup['latitude'].tolist()
fig = plt.figure(figsize=(30,30))
ax=fig.add_subplot(1,1,1)
ax = m.drawcountries()
ax = m.drawcoastlines()
ax = m.fillcontinents(color='#eadf93')
ax = m.drawmapboundary()
ax = m.scatter(longitudes, latitudes, s=80, zorder=2, latlon=True, color='#0e3bef')


# Regards,
# Maaz Imran
