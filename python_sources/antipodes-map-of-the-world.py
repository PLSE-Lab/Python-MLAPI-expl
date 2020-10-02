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


# **What are Antipodes?**
# 
# In geography, the antipode of any spot on Earth is the point on Earth's surface diametrically opposite to it.
# 
# A pair of points antipodal to each other are situated such that a straight line connecting the two would pass through Earth's center.
# 
# Antipodal points are as far away from each other as possible
# 
# Approximately 15% of land territory is antipodal to other land, representing approximately 4.4% of the Earth's surface.

# Import Statements

# In[ ]:


#Visualization
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# Read Data

# In[ ]:


data=pd.read_csv('../input/world-cities-database/worldcities.csv')
data.head()


# In[ ]:


plt.scatter(data.lng,data.lat)


# **Taking relevant data from total data.**
# 
# For plotting antipode map, we only require latitue and longitude

# In[ ]:


city=data[['lat','lng']]
city.head()


# **Splitting into North and South Hemispheres**

# In[ ]:


city_N=city[city.lat>0]      #North  
city_S=city[city.lat<0]      #South

plt.scatter(city_N.lng,city_N.lat)
plt.scatter(city_S.lng,city_S.lat)


# **Transform Map of Southern Hemishphere**
# 
# 
# * For a city in equator(0 degree latitude), its antipode lies on equator
# * For 20 S (-20) antipode in 20 N
# * For 60 S (-60) antipode in 60 N, and so forth
# 
# **Hence, we observe that latitude transformation can be done by taking absolute value.**
# 

# In[ ]:


city_S.lat=abs(city_S.lat)
plt.scatter(city_S.lng,city_S.lat)


# **Longitude transformation:**
# 
# Longitude transformation is more tricky, having completed the latitude transformation, we will just think about points in equator.
# 
# The eastern most point(180) is opposite 0, as we move towards west(towards 0), the antipode also moves towards west(less than 0).
# 
#     180 corresponds to 0
#     90 corresponds to -90
#     0 corresponds to -180
# 
# Hence we can say that for an **Easten city with longitude x, antipode is in x-180.**
# 
# Similarly in the western half also we can see that as we move from western most point(-180) to 0, we can observe that the antipode moves from 0 to 180.
# 
#     -180 corresponds to 0
#     -90 corresponds to 90
#     0 corresponds to 180
# 
# Hence we can say that for a** Western city with longitude x, antipode is in x+180.**

# In[ ]:


city_SE=city_S[city_S.lng>0] # South East
city_SW=city_S[city_S.lng<0] # South West

city_SE.lng=city_SE.lng-180
city_SW.lng=city_SW.lng+180

# Transfromed map of Southern hemisphere
city_ST=pd.concat([city_SE,city_SW], axis=0)

plt.scatter(city_ST.lng,city_ST.lat)


# ***Overlay Transformed map of Southern Hemisphere over Northern Hemisphere map to get Antipode Map***

# In[ ]:


plt.figure(figsize=(16,10))
plt.title("Antipode Map of the world", fontsize=18)
plt.scatter(city_N.lng,city_N.lat, alpha=0.2)
plt.scatter(city_ST.lng,city_ST.lat, alpha=0.2)


# In[ ]:




