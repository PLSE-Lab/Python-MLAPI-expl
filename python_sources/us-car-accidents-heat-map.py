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


import folium
from folium.plugins import HeatMap
import matplotlib as plt


# In[ ]:


df = pd.read_csv("/kaggle/input/us-accidents/US_Accidents_Dec19.csv")


# In[ ]:


df.head()


# In[ ]:


df = df[["Severity", "Start_Lat", "Start_Lng"]]


# In[ ]:


df.shape


# In[ ]:



df3 = df[df['Severity'] > 3]  


# In[ ]:


df3.shape


# In[ ]:


df3.head()


# In[ ]:


severities = df[['Severity']]


# In[ ]:


severities.head()


# In[ ]:


df3 = df3[["Start_Lat", "Start_Lng"]]


# In[ ]:


map_van = folium.Map(location = [37.773972, -122.431297], zoom_start = 13)

HeatMap(df3).add_to(map_van)


# In[ ]:


map_van


# In[ ]:


df.head()


# In[ ]:


df4 = df[df["Severity"]>3]


# In[ ]:


df4.shape


# In[ ]:


df4.head()


# In[ ]:


df4 = df4[["Start_Lat", "Start_Lng"]]


# In[ ]:


map_van = folium.Map(location = [37.773972, -122.431297], zoom_start = 13)

HeatMap(df4).add_to(map_van)


# In[ ]:


map_van


# In[ ]:




