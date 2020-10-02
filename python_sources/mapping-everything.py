#!/usr/bin/env python
# coding: utf-8

# # Mapping The Cost Of Living
# 
# <img src="https://i.imgur.com/YBH4F2W.jpg" width="1000px">
# 
# The data for this has been pulled from [Numbeo](https://www.numbeo.com/cost-of-living/prices_by_city.jsp?itemId=101&itemId=100&itemId=228&itemId=224&itemId=60&itemId=66&itemId=64&itemId=62&itemId=110&itemId=118&itemId=121&itemId=14&itemId=19&itemId=17&itemId=15&itemId=11&itemId=16&itemId=113&itemId=9&itemId=12&itemId=8&itemId=119&itemId=111&itemId=112&itemId=115&itemId=116&itemId=13&itemId=27&itemId=26&itemId=29&itemId=28&itemId=114&itemId=6&itemId=4&itemId=5&itemId=3&itemId=2&itemId=1&itemId=7&itemId=105&itemId=106&itemId=44&itemId=40&itemId=42&itemId=24&itemId=20&itemId=18&itemId=109&itemId=108&itemId=107&itemId=206&itemId=25&itemId=32&itemId=30&itemId=33&displayCurrency=USD) and looks to be priced in Euros.
# 
# It lists a range of costs across world cities, ranging from the price for a kilogram of Apples, to how much for accomodation.
# 
# First I've used folium to map each city across a few prices. OpenCage API was used to retrieve the latitude and longitude.
# 
# Then I've grouped the mean prices for all the countries and created a choropleth with matplotlib for EVERYTHING.
# 
# I plan to do a lot more. If you upvote I'll really appreciate it.
# 
# * [Using OpenCage to Retrieve Latitude and Longitude](#opencage)
# * [Mapping with Folium](#folium)
# * [Mapping a Choropleth with MatPlotLib](#choropleth)
# * [Mapping All Columns Normalized](#normalize)
# * [Further Analysis](#furtheranalysis)

# In[ ]:


import pandas as pd
import geopandas as gpd
import numpy as np
import folium
from folium import Circle
import matplotlib.pyplot as plt
from sklearn import preprocessing

#Function for displaying the map
#def embed_map(m, file_name):
#    from IPython.display import IFrame
#    m.save(file_name)
#    return IFrame(file_name, width='100%', height='500px')


# In[ ]:


# data needs to be transformed so we have the rows are cities.
df = pd.read_csv("/kaggle/input/cost-of-living/cost-of-living.csv", index_col=[0]).T.reset_index()
df = df.rename(columns={'index':'location'})
df.head()


# <a id='opencage'></a>
# ## Using OpenCage to Retrieve Latitude and Longitude

# In[ ]:


get_ipython().system('pip install opencage')


# In[ ]:


from opencage.geocoder import OpenCageGeocode
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
key = user_secrets.get_secret("key")

geocoder = OpenCageGeocode(key)

list_lat = [] 
list_long = []

for row in df.location:
    try:
        query = str(row)
        results = geocoder.geocode(query)   
        lat = results[0]['geometry']['lat']
        long = results[0]['geometry']['lng']
        list_lat.append(lat)
        list_long.append(long)
    except:
        list_lat.append(None)
        list_long.append(None)

df['lat'] = list_lat   
df['lon'] = list_long


# In[ ]:


df['city'] = df['location'].apply(lambda x: str(x).split(', ')[0])


# <a id='folium'></a>
# ## Mapping with Folium

# In[ ]:


# To find some interesting columns to plot I've sorted them by range. 
# Perhaps a better way to do this in future would be by variance.
top_range = (df.describe().loc['min',:]/df.describe().loc['max',:]).sort_values().index[2:22]
list(top_range)


# In[ ]:


def color_producer(val):
    if val <= df[item].quantile(.25):
        return 'forestgreen'
    elif val <= df[item].quantile(.50):
        return 'goldenrod'
    elif val <= df[item].quantile(.75):
        return 'darkred'
    else:
        return 'black'


# In[ ]:


m_1 = folium.Map(location=[df.lat.mean(),df.lon.mean()], tiles='cartodbpositron', zoom_start=2)

item = top_range[0]

# Add a bubble map to the base map
for i in range(0,len(df)):
    Circle(
        location=[df.iloc[i]['lat'], df.iloc[i]['lon']],
        radius=1000,
        color=color_producer(df.iloc[i][item])).add_to(m_1)

print ('Price of: ', item)
m_1


# In[ ]:


m_2= folium.Map(location=[df.lat.mean(),df.lon.mean()], tiles='cartodbpositron', zoom_start=2)
item = top_range[2]

# Add a bubble map to the base map
for i in range(0,len(df)):
    Circle(
        location=[df.iloc[i]['lat'], df.iloc[i]['lon']],
        radius=1000,
        color=color_producer(df.iloc[i][item])).add_to(m_2)

print ('Price of: ', item)
# Display the map
#e
m_2


# In[ ]:


m_3= folium.Map(location=[df.lat.mean(),df.lon.mean()], tiles='cartodbpositron', zoom_start=2)
item = top_range[9]

# Add a bubble map to the base map
for i in range(0,len(df)):
    Circle(
        location=[df.iloc[i]['lat'], df.iloc[i]['lon']],
        radius=1000,
        color=color_producer(df.iloc[i][item])).add_to(m_3)

print ('Price of: ', item)
# Display the map
#e
m_3


# In[ ]:


m_4= folium.Map(location=[df.lat.mean(),df.lon.mean()], tiles='cartodbpositron', zoom_start=2)
item = 'Toyota Corolla 1.6l 97kW Comfort (Or Equivalent New Car)'

# Add a bubble map to the base map
for i in range(0,len(df)):
    Circle(
        location=[df.iloc[i]['lat'], df.iloc[i]['lon']],
        radius=1000,
        color=color_producer(df.iloc[i][item])).add_to(m_4)

print ('Price of: ', item)
m_4


# <a id='choropleth'></a>
# ## Mapping a Choropleth with MatPlotLib

# Later I'd like to expore Choropleth mapping with Folium and perhaps Bokeh, but here I plan to map every column in the original dataframe, and it'll be much faster and easier to just use MatPlotLib.
# 
# I'm going to group the cities by country using the mean of all the columns. This will give a much clearer overview when looking at the map, when trying to detect any trends.

# In[ ]:


# I spent sometime trying to set up another API using the Overpass Turbo API with OpenStreetMap for geometry data.
# Fortuantely I found Geopandas already has everything ready to import.
cities = gpd.read_file(gpd.datasets.get_path('naturalearth_cities'))
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))


# In[ ]:


df['country'] = df.location.apply(lambda x: str(x).split(', ')[-1])
countries = df.groupby('country', as_index=False).mean()


# In[ ]:


name_change = {'Bosnia And Herzegovina' : 'Bosnia and Herz.',
'United States' : 'United States of America',
'Czech Republic' : 'Czechia',
'Dominican Republic' : 'Dominican Rep.'}

countries['country'] = countries.country.replace(name_change)


# In[ ]:


world = world[world.name.isin(countries.country.values)]
world = world.sort_values(by='name').reset_index()
countries = countries.sort_values(by='country').reset_index()
world = world.merge(countries, left_on=['name'], right_on=['country'])


# In[ ]:


prices = countries.columns[2:-2]
fig, ax = plt.subplots(len(prices), figsize=(16,6*len(prices)))

c = 0
for i in range(len(prices)):
    
    # some column names are repeated in the dataset, but the data is different.
    # An if-else makes sure each of these repeated columns in mapped.
    if type(world[prices[i]]) is pd.DataFrame:
        col = world[prices[i]].iloc[:,c]
        c -= 1
        c = abs(c)
    else:
        col = world[prices[i]] 
                              
    world.plot(column=col,
                ax=ax[i],
                legend=True,
                legend_kwds={'label': "Cost"})
    ax[i].title.set_text(prices[i])


# <a id='normalize'></a>
# ## Mapping All Columns Normalized

# This will give a great overview of which areas are the most and least expensive.

# In[ ]:


data = world.iloc[:,9:]
x = data.values
min_max_scalar = preprocessing.MinMaxScaler()
x_scaled = min_max_scalar.fit_transform(x)
data_norm = pd.DataFrame(x_scaled)
data_norm.columns = data.columns


# In[ ]:


df_summary = pd.DataFrame(world['country'])
df_summary['total'] = data_norm.iloc[:,:56].mean(axis=1)


# In[ ]:


fig, ax = plt.subplots(1, figsize=(16,6))
                              
world.plot(column=df_summary['total'], ax=ax,
            legend=True,
            legend_kwds={'label': "Most to least expensive place to live"})
ax.title.set_text("All prices normalized for each Country")


# It looks like Switzerland, Iceland, and Norway are the most expensive of places. This can be confirmed by looking at the data below.
# 
# It's also clear that Europe and North America are some of the most expensive places on Earth.

# In[ ]:


df_summary.sort_values(by='total', ascending=False).head(3)


# <a id='furtheranalysis'></a>
# ## Further Analysis
# 
# Some further areas I would like to expore in a future notebook would be;
# * Grouping columns by similarity. Which places are more expensive for food vs. accomodation.
# * Outliers. Which places are much cheaper or expensive for particular things, and exploring possible reasons for these.
# * Exploring the correlation between how much things cost and their countries GDP.
