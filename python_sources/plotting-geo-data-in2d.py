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


# In[ ]:


import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString,LinearRing
import shapely.wkt


# In[ ]:


import pandas as pd
plot = pd.read_csv("../input/finaltofinal.csv")


# In[ ]:


geometry = [Point(xy) for xy in zip(plot.x, plot.y)]
geoData = gpd.GeoDataFrame(plot, geometry=geometry)

geoData1 = geoData.groupby(['sid'])['geometry'].apply(lambda x: LineString(x.tolist()))
geoData2= gpd.GeoDataFrame(geoData1, geometry='geometry')


# In[ ]:


geoData2.crs = {'init': 'epsg:4326'}


# In[ ]:


geoData2.head()


# In[ ]:


geoData2.plot()

