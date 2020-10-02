#!/usr/bin/env python
# coding: utf-8

# Print a world map from the logged ship locations inspired by [Ben Schmidt's ship log maps](http://sappingattention.blogspot.com/2012/11/reading-digital-sources-case-study-in.html).

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


# Input data files are available in the "../input/" directory.
get_ipython().system('ls ../input')


# In[ ]:


df = pd.read_csv('../input/CLIWOC15.csv', usecols=['Lat3', 'Lon3', 'Nationality'])
df.dropna(axis=0, how='any', inplace=True)
df.shape


# In[ ]:


fig = plt.figure(figsize=(20, 14))
markersize = .5
markertype = '.'
markercolor = '#000000'
markeralpha = .4

m = Basemap(projection='mill')

# Avoid border around map.
m.drawmapboundary(fill_color='#ffffff', linewidth=.0)

# Convert locations to x/y coordinates and plot them as dots.
x, y = m(df.Lon3.values, df.Lat3.values)
m.scatter(x, y, markersize, marker=markertype, color=markercolor, alpha=markeralpha)
plt.show()

