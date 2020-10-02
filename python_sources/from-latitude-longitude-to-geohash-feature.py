#!/usr/bin/env python
# coding: utf-8

# # Geohashing
# 
# I want to share with you a trick I have used in the past to work with latitude/longitude data in ML models: the idea is to take the raw values and discretize into tiles using a geohashing technique (google s2, uber h3, as well as the original geohash). Once a tileID is generated out of the raw latitude/longitude, it can then be mapped into a OHE variable (it would be quite high cardinality though). 
# 
# Let me show you some maps to give you an idea...

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


get_ipython().system('pip install h3')


# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


DATA_FOLDER = '/kaggle/input/bigquery-geotab-intersection-congestion/'


# In[ ]:


train_data = pd.read_csv(DATA_FOLDER + 'train.csv')
test_data = pd.read_csv(DATA_FOLDER + 'test.csv')

print(train_data.shape)
print(test_data.shape)


# In[ ]:


import folium

from h3 import h3


# In[ ]:


COLOR_MAP = {
    1: 'orange',
    2: 'red',
    3: 'green'
}

CITY_2_GEO = {
    'Atlanta': [33.759004, -84.389609],
    'Philadelphia': [39.952778, -75.163611],
    'Boston': [42.358056, -71.063611],
    'Chicago': [41.881944, -87.627778]
}

HEX_ADDR_PRECISION = 7


# In[ ]:


def plot_on_map(train_df, test_df, city = 'Atlanta'):
    curr_map = folium.Map(location=CITY_2_GEO[city], 
                   zoom_start=11, 
                   prefer_canvas=True, 
                   tiles='stamentoner')
    
    city_train_df = train_df[train_df['City'] == city]
    city_test_df = test_df[test_df['City'] == city]
    
    city_train_df = city_train_df[['IntersectionId', 'Latitude', 'Longitude']].drop_duplicates()
    city_train_df['Group'] = 1

    city_test_df = city_test_df[['IntersectionId', 'Latitude', 'Longitude']].drop_duplicates()
    city_test_df['Group'] = 2

    df1 = city_train_df.append(city_test_df)

    points = df1.groupby(['IntersectionId', 'Latitude', 'Longitude']).sum().reset_index()
    for index, row in points.iterrows():
        folium.CircleMarker([row['Latitude'], row['Longitude']],
                            radius=5,
                            popup=row['IntersectionId'],
                            color=COLOR_MAP[row['Group']],
                           ).add_to(curr_map)

    # build hex from train dataset
    geo_unique = city_train_df[['Latitude', 'Longitude']].drop_duplicates()
    geo_unique['hex_addr'] = np.vectorize(lambda longitude, latitude: h3.geo_to_h3(latitude, longitude, HEX_ADDR_PRECISION))(geo_unique['Longitude'], geo_unique['Latitude'])
    
    for hex_addr in geo_unique['hex_addr'].unique():       
        polygons = h3.h3_set_to_multi_polygon([hex_addr], geo_json=False)
        outlines = [loop for polygon in polygons for loop in polygon]
        polyline = [outline + [outline[0]] for outline in outlines][0]
        
        folium.PolyLine(locations=polyline,
                        weight=8,
                        color='yellow',
                        opacity=0.15,
                        fill_color='yellow',
                        fill_opacity=0.05,
                        fill=True).add_to(curr_map)
    
    return curr_map


# Dots indicates the intersections in the provided datasets, with:
# - Green: available in both datasets
# - Orange: available only in the *train* dataset
# - Red: available only in the *test* dataset

# In[ ]:


plot_on_map(train_data, test_data, 'Atlanta')


# In[ ]:


plot_on_map(train_data, test_data, 'Philadelphia')


# In[ ]:


plot_on_map(train_data, test_data, 'Boston')


# In[ ]:


plot_on_map(train_data, test_data, 'Chicago')

