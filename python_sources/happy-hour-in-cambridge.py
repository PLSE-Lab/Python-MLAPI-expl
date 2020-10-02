#!/usr/bin/env python
# coding: utf-8

# # Happy hour's Kaggle in Cambridge
# Thank you for updating an interesting dataset in Kaggle.
# In order to keep my motivation, let me try to check the streets for my happy-hour ;)

# In[ ]:


# OK, let me call my friends
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir('../input/'))


# In[ ]:


data = pd.read_csv('../input/open_pubs.csv')


# In[ ]:


# Check the feeling
data.head()


# In[ ]:


data.drop(data.index[data.latitude == '\\N'], inplace = True)
data.drop(data.index[data.longitude == '\\N'], inplace = True)
data.drop(data.index[data.longitude == 'Broxbourne'], inplace = True)
data.drop(data.index[data.longitude == 'Ryedale'], inplace = True)


# In[ ]:


data.info()


# **Yey, Great!!** It seems easy.

# In[ ]:


data['latitude_float'] = data['latitude'].astype(np.float64)
data['longitude_float'] = data['longitude'].astype(np.float64)


# Just one point.
# 'Latitude' and 'Logitude' were converted to num for inputting folium.

# In[ ]:


data.head(5)


# OK. Let's see good streets.

# In[ ]:


import folium
map = folium.Map(
    location=[52.204990, 0.122139]
    , tiles = 'OpenStreetMap'
    , zoom_start = 9
)

for each in data[:1000].iterrows():
    folium.CircleMarker([each[1]['latitude_float'],
                         each[1]['longitude_float']],
                        radius = 5,
                        color = 'blue',
                        popup = str(each[1]['name']) + '\\n' + str(each[1]['address']),
                        fill_color = '#FD8A6C'
                       ).add_to(map)

map


# **Yey!!** How do you think which road is most interesting?
# 
# ### Thanks again for your updating this data. Have your good beer ;)
