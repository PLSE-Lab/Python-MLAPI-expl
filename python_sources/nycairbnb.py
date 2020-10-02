#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns


# In[ ]:


data = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv', encoding='utf=8')
data.shape


# In[ ]:


data.head(3)

Here goes our dataset. Before moving on, let's check some of the columns, say, neighbourhood_group, about what values they take.
# In[ ]:


set(data['neighbourhood_group'])


# In[ ]:


set(data['room_type'])


# Alright. We have some sense about their basic classifications. There is one column with relatively continuous numerical value, which is price. We'd better see its overall distribution.

# In[ ]:


plt.figure(figsize=(14,8))
plt.title("Price Distribution")
sns.distplot(data['price'])


# We can see en enormously long tail. There's gotta be some outliars, which we don't want for now.

# In[ ]:


data['price'].describe(percentiles = [0.2*x for x in range(5)])


# We see that most values are located under $200, which is still a big value for one night in airbnb, and there is this outliar of 10000, which I believe should be a mistaken figure. We'll leave those out anyways.

# In[ ]:


before = len(data)
data = data[data['price']<=500]
after = len(data)
print("With price ceiling $500, {0} values were omitted, leaving {1}".format(before-after, after))


# Since I had my price ceiling at $500, 1044 rows were dropped, which I think is tolerable. We've still got over 45K rows anyways. Now it's time for another distplot.

# In[ ]:


plt.figure(figsize=(14,8))
plt.title("Price Distribution - renewed")
sns.distplot(data['price'])


# Here we see a better picture. Most prices lie around $80~$100, so the graph is little skewed to the right. 

# In[ ]:


data['price'].describe(percentiles=[0.25*x for x in range(4)])


# In[ ]:


data.columns


# In[ ]:


data.groupby('neighbourhood_group')['price'].mean()


# In[ ]:


val = data.groupby('neighbourhood_group')['price'].mean()
ind = val.index

plt.figure(figsize=(13,7))
plt.bar(ind, val, color='khaki', edgecolor='k')
plt.title("AVG price among neighborhoods")
plt.ylabel('price($)')


# Obviously, Manhattan turns out to be the most pricy place to spend a night. Let's see a little more.

# In[ ]:


manhattan = data[data['neighbourhood_group'] == 'Manhattan']
manhattan.sort_values(by='price', ascending=False).iloc[:5]


# In[ ]:


manhattan.groupby('neighbourhood')['price'].mean().sort_values(ascending=False)[:10]


# In[ ]:


manhattan.groupby('neighbourhood')['price'].mean().sort_values(ascending=True)[:10]


# Since in real estate everything is about location, I want to see a map that shows their geographical distributions. It's my first time using lat, long coordinates to plot geographical information, so I'll try numerous libraries available for free.

# In[ ]:


from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame

geometry = [Point(xy) for xy in zip(manhattan['longitude'], manhattan['latitude'])]
gdf = GeoDataFrame(manhattan, geometry=geometry)   

#this is a simple map that goes with geopandas
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
gdf.plot(ax=world.plot(figsize=(14, 8)), marker='^', color='red', markersize=15)
plt.xlim(manhattan.longitude.min()-5, manhattan.longitude.max()+5)
plt.ylim(manhattan.latitude.min()-5, manhattan.latitude.max()+5)


# In[ ]:




