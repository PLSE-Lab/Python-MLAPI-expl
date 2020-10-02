#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


pot_holes_df= pd.read_csv('../input/311-service-requests-pot-holes-reported.csv')
pot_holes_df.head()


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


# No of Potholes in each Zip code
pot_holes_df.groupby('ZIP')['CREATION DATE'].count().plot(kind='bar',figsize=(15, 6))
plt.show()
# No of Potholes in each Ward
pot_holes_df.groupby('Ward')['CREATION DATE'].count().plot(kind='bar',figsize=(15, 6))
plt.show()
# No of Potholes in Police District
pot_holes_df.groupby('Police District')['CREATION DATE'].count().plot(kind='bar',figsize=(15, 6))
plt.show()
# No of Potholes in Community Area
pot_holes_df.groupby('Community Area')['CREATION DATE'].count().plot(kind='bar',figsize=(15, 6))
plt.show()


# In[ ]:


import folium


# In[ ]:


pot_holes_df_cleaned=pot_holes_df.dropna()


# In[ ]:


pot_holes_df_cleaned.head()


# In[ ]:


location = pot_holes_df_cleaned['LATITUDE'].mean(), pot_holes_df_cleaned['LONGITUDE'].mean()
location


# In[ ]:


len(pot_holes_df_cleaned)


# In[ ]:


pothole_map = folium.Map(location=location, height = 600, tiles='OpenStreetMap', zoom_start=13)
for i in range(0,100):
  folium.Marker([float(pot_holes_df_cleaned.iloc[i]['LATITUDE']), float(pot_holes_df_cleaned.iloc[i]['LONGITUDE'])], icon=folium.Icon(color="red")).add_to(pothole_map)
pothole_map

