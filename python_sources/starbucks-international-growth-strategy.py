#!/usr/bin/env python
# coding: utf-8

# Background
# ----------
# 
# Retail chains generally have an international growth strategy based on the target country's laws, regulations, culture, economics, politics, etc. influencing decision like - for example - how they open new stores. In this notebook I try to find the strategy that Starbucks has counted on for their international growth plans, with respect to the ownership structure of their stores worldwide.

# In[ ]:


# We will mainly explore through visualizations
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.patches as mpatches

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Reading-in the data
starbucks = pd.read_csv('../input/directory.csv')
starbucks.columns


# ## Number of stores ##

# In[ ]:


starbucks.shape[0]


# Missing data
# --------------

# In[ ]:


starbucks.isnull().sum()


# We seem to be okay for our little analysis.

# ## Top 10 city locations for Starbucks stores ##

# In[ ]:


starbucks['City'].value_counts().head(10)


# That is surprising to me. Shanghai has almost twice as many stores as New York?? I live in New York and I have lived in Shanghai, and it certainly didn't seem so!

# ## How is the ownership structured for stores ##

# In[ ]:


starbucks['Ownership Type'].value_counts()


# In[ ]:


sns.set(style="whitegrid", context="talk")
sns.countplot(x='Ownership Type', data=starbucks, palette="BuGn_d")


# So, most Starbucks stores are either owned by the company or licensed. Got it.

# Understanding the Starbucks strategy for store ownership
# --------------------------------------------------------
# 
# Unfortunately, to visualize the stores on a world map and color them based on ownership-type, we need to loop through every single store and plot it individually. 
# 

# In[ ]:


f, ax = plt.subplots(figsize=(18,12))
map = Basemap(projection='mill', 
              llcrnrlat = -80,
              urcrnrlat = 80,
              llcrnrlon = -180,
              urcrnrlon = 180,
              resolution = 'h')

# Plot coastlines and country borders.
map.drawcoastlines()
map.drawcountries()

# Fill continents and color wet areas.
map.drawmapboundary(fill_color='lightskyblue')
map.fillcontinents(color='beige',
                   lake_color='lightskyblue')

# Color coding the store by ownership type.
markerCode = {'Company Owned': 'green', 
              'Licensed': 'yellow',
              'Franchise': 'tomato',
              'Joint Venture': 'mediumpurple'}
starbucks['ownerColorCode'] = starbucks['Ownership Type'].map(markerCode)


lons, lats = map(list(starbucks["Longitude"].astype(float)),
                 list(starbucks["Latitude"].astype(float)))
colors = list(starbucks['ownerColorCode'])

# Loop through each location to plot the individual stores.
for lon, lat, owner in zip(lons, lats, colors):
    x, y = lon, lat
    marker_string = owner
    map.plot(x, y, 'bo', alpha = 0.6, color = marker_string)

# Drop the color-code column after use.
starbucks.drop(['ownerColorCode'], axis=1, inplace=True)

# Hack together a legend
legend_handles = [mpatches.Patch(color = color_code, label = owner) for owner, color_code in markerCode.items()]
ax.legend(loc='lower left',
          handles = legend_handles)

plt.title('Starbucks Stores and Ownership')
plt.show()


# As we suspected, country regions have clusters of same store-ownership types. For example, all the stores in India are Joint Ventures. China has a lot of company-owned stores, which incidentally was the result of a change in Starbucks' strategy in China a few years ago. They later did a similar change from JVs to full ownership in Japan as well, which is why we see a mix of purple and green dots in the Japan region. 
# 
# 
# 
# Though its hard to see whats going on in Europe. Lets zoom in...

# In[ ]:


# A long list of countries I think are in Europe.
europe =['BE', 'EL', 'LT', 'PT', 'BG', 'ES', 'LU', 'RO', 'CZ', 'FR', 'HU', 'SI', 'DK', 'GB', 'GR', 'HR', 'MT', 'SK', 'DE', 'IT', 'NL', 'FI', 'EE', 'CY', 'AT', 'SE', 'IE', 'LV', 'PL', 'UK', 'IS', 'NO', 'LI', 'CH', 'TR']
european_stores = starbucks[starbucks['Country'].isin(europe)]

f, ax = plt.subplots(figsize=(18,12))

# Where I think Europe is...
map = Basemap(projection='mill', 
              llcrnrlat = 30,
              urcrnrlat = 70,
              llcrnrlon = -20,
              urcrnrlon = 40,
              resolution = 'l')

# Plot coastlines and country borders.
map.drawcoastlines()
map.drawcountries()

# Fill continents and color wet areas.
map.drawmapboundary(fill_color='lightskyblue')
map.fillcontinents(color='beige',
                   lake_color='lightskyblue')

# Color coding the store by ownership type.
markerCode = {'Company Owned': 'green', 
              'Licensed': 'yellow',
              'Franchise': 'tomato',
              'Joint Venture': 'mediumpurple'}
european_stores['ownerColorCode'] = european_stores['Ownership Type'].map(markerCode)


lons, lats = map(list(european_stores["Longitude"].astype(float)),
                 list(european_stores["Latitude"].astype(float)))
colors = list(european_stores['ownerColorCode'])

# Loop through each location to plot the individual stores.
for lon, lat, owner in zip(lons, lats, colors):
    x, y = lon, lat
    marker_string = owner
    map.plot(x, y, 'bo', alpha = 0.6, color = marker_string)

# Drop the color-code column after use.
european_stores.drop(['ownerColorCode'], axis=1, inplace=True)

# Hack together a legend
legend_handles = [mpatches.Patch(color = color_code, label = owner) for owner, color_code in markerCode.items()]
ax.legend(loc='lower left',
          handles = legend_handles)

plt.title('Starbucks Stores and Ownership in Europe')
plt.show()


# Starbucks seems popular in U.K., which can be expected as it is pretty close to the U.S. in terms of political, social and cultural distance. Although most stores are either franchises or licensed stores. France has a lot of franchised stores as well, whereas Germany has mostly company-owned stores, Poland and Czech Republic have joint-ventures, and Spain, Turkey, Ireland, Greece, Bulgaria, Romania, Norway, Sweden have all licensed stores. 
# 
# Interesting to note that Italy - the country of coffee - has no Starbucks stores, mainly explained by their strong tradition of the local coffee stores which has been a challenge for Starbucks. Starbucks recently announced a partnership to open its first licensed store in Milan.

# ## References ##
# 
# - [Starbucks announces it will open stores in Italy in 2017][1]
# - [Starbucks' full ownership of Japan unit to boost international growth][2]
# - [Starbucks acquires full ownership of Starbucks stores in Central, South and Western China][3]
# 
# 
#   [1]: https://news.starbucks.com/news/starbucks-announces-it-will-open-stores-in-Italy-in-2017
#   [2]: https://www.forbes.com/sites/greatspeculations/2014/09/30/starbucks-full-ownership-of-japan-unit-to-boost-international-revenue-growth/#1cf971eb428b
#   [3]: http://www.restaurantnews.com/starbucks-acquires-full-ownership-of-starbucks-stores-in-central-south-western-china/
