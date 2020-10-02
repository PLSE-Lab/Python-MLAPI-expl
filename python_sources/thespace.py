#!/usr/bin/env python
# coding: utf-8

# # The Space
# 
# In this script we explore the area and dimentions of places. 
# We also see that the space is anisotropic, variance along x about 33 time more than y. 
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt


# In[ ]:


train = pd.read_csv('../input/train.csv', index_col='row_id')


# In[ ]:


plt.rcParams['figure.figsize'] = (12,10)
train.sample(n=1000000).plot(x='x', y='y', kind='hexbin', gridsize=100)
plt.title('Density of checkins')
plt.xlabel('x')


# In[ ]:


train.x.hist(bins=1000)


# ## Radius of Gyration and Area
# 
# They tells us how big a place is.

# In[ ]:


g = train.groupby('place_id')
place = g.mean()

place['counts'] = g.x.count()

std = g.std()
mean = g.mean()
place['rg'] = np.sqrt(std.x**2 + std.y**2)
place['area'] = std.x * std.y


# In[ ]:


plt.rcParams['figure.figsize'] = (16,4)
place.rg.hist(bins=200, log=True);
plt.xlabel('Radius of Gyration')


# In[ ]:


place.area.hist(bins=200, log=True);
plt.xlabel('Area')


# Some places are really big.  Let zoom in:

# In[ ]:


min_count = 1500
print(place.rg[place.counts[place.counts > min_count].index].max())
print(place.rg[place.counts[place.counts > min_count].index].min())
largest_place = place.rg[place.counts[place.counts > min_count].index].argmax()
smallest_place = place.rg[place.counts[place.counts > min_count].index].argmin()


# In[ ]:


plt.rcParams['figure.figsize'] = (4,4)

train.ix[(train.place_id == largest_place),:].plot(x='x', y='y', kind='scatter')
plt.title('largest rg')
plt.xlim(0,10)
plt.ylim(0,10);


# In[ ]:


train.ix[(train.place_id == smallest_place),: ].plot(x='x', y='y', kind='scatter')
plt.title('smallest rg')
plt.xlim(0,10)
plt.ylim(0,10);


# This is a place with more than 1500 checkins. 
# y goes from 0.31 to 0.46, but x goes from **0.02** to **9.3**. 
# That should have been a very long place!

# In[ ]:


min_count = 1000
print(place.area[place.counts[place.counts > min_count].index].max())
print(place.area[place.counts[place.counts > min_count].index].min())
largest_area = place.area[place.counts[place.counts > min_count].index].argmax()
smallest_area = place.area[place.counts[place.counts > min_count].index].argmin()


# In[ ]:


plt.rcParams['figure.figsize'] = (4,4)
train[train.place_id == largest_area].plot(x='x', y='y', kind='scatter')
plt.title('largest area')

plt.xlim(0,10)
plt.ylim(0,10);


# In[ ]:


plt.rcParams['figure.figsize'] = (4,4)
train[train.place_id == smallest_area].plot(x='x', y='y', kind='scatter')
plt.title('smallest area')

plt.xlim(0,10)
plt.ylim(0,10);


# ## Anisotropy of the space
# 
# The standard deviation of places along x is 32.7 time more than y:

# In[ ]:


std.x.mean()/std.y.mean()

