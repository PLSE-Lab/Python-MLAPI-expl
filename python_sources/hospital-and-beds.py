#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## import libraries
import pandas as pd
import numpy as np
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

import json

import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# read data
# data about number of goverment hospitals and beds in hospital
df = pd.read_csv('/kaggle/input/hospitals-and-beds-in-india/Number of Government Hospitals and Beds in Rural and Urban Areas .csv')


# In[ ]:


df.head()


# In[ ]:


print('The shape of data set :', df.shape)


# In[ ]:


# change column names
df.rename (columns = {'Unnamed: 2': 'Beds in Rural', 'Unnamed: 4':'Beds in Urban', 'As on' : 'As on date' , 'States/UTs':'State'}, inplace = True)
# drop first row
df = df.drop(0)


# In[ ]:


df = df.drop(37)


# In[ ]:


# remove * character from state feature
df['State']= df['State'].str.replace('*','')


# In[ ]:


#convert data types of numerical values into int
df[['Rural hospitals','Beds in Rural','Urban hospitals','Beds in Urban']]=df[['Rural hospitals','Beds in Rural','Urban hospitals','Beds in Urban']].astype(int) 


# In[ ]:


# total numbers of beds
df['Total beds'] = df['Beds in Rural'] + df['Beds in Urban']
df['Total Hospital'] = df['Rural hospitals'] + df['Urban hospitals']


# In[ ]:


df.head()


# In[ ]:


df.describe()


# In[ ]:


df.info()


# In[ ]:


## to draw choropleth map 
state_data = df[['State','Total Hospital','Total beds']]


# In[ ]:


## we need same state name in both data and json file
state_data['State'] = state_data['State'].str.replace('Jammu & Kashmir','Jammu and Kashmir')
state_data['State'] = state_data['State'].str.replace('Odisha','Orissa')
state_data['State'] = state_data['State'].str.replace('Uttarakhand','Uttaranchal')


# In[ ]:


# dropping union teritorry cause they are not avaliable in json file
state_data.drop(index = [30,31,32,33,36,35],inplace = True)


# In[ ]:


import folium
import json


# In[ ]:


india_map = folium.Map(location=[20.5937, 78.9629], zoom_start=5)

state_geo = os.path.join('/kaggle/input/indian-state-json/states2.json')

folium.Choropleth(
    geo_data=state_geo,
    name='choropleth',
    data=state_data,
    columns=['State', 'Total Hospital'],
    key_on='feature.id',
    fill_color='YlGn',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Total gov. Hospital'
).add_to(india_map)

folium.LayerControl().add_to(india_map)


# In[ ]:


## choropleth map for total gov. hospital in india
india_map


# In[ ]:


## choropleth map for total no. beds in gov. hospital

india_map2 = folium.Map(location=[20.5937, 78.9629], zoom_start=5)

state_geo = os.path.join('/kaggle/input/indian-state-json/states2.json')

folium.Choropleth(
    geo_data=state_geo,
    name='choropleth',
    data=state_data,
    columns=['State', 'Total beds'],
    key_on='feature.id',
    fill_color='OrRd',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Total Beds in gov. hospital'
).add_to(india_map2)

folium.LayerControl().add_to(india_map)


# In[ ]:


india_map2


# In[ ]:


plt.figure(figsize=(10,10))
plt.barh(df['State'],df['Rural hospitals'], label = 'Rural Hospitals')
plt.barh(df['State'],df['Urban hospitals'], label = 'Urban Hospitals')
plt.ylabel('States')
plt.xlabel('No. of hospital')
plt.legend()
plt.show()


# In[ ]:


plt.figure(figsize=(10,10))
plt.barh(df['State'],df['Beds in Rural'], label = 'Beds in Rural')
plt.barh(df['State'],df['Beds in Urban'], label = 'Beds in Urban')
plt.ylabel('States')
plt.xlabel('No. of beds')
plt.legend()
plt.show()


# In[ ]:


plt.figure(figsize=(10,10))
sns.boxplot( df['Rural hospitals'], df['State'])
plt.show()


# In[ ]:


# Hospitals and beds maintained by Railways
df2 = pd.read_csv('/kaggle/input/hospitals-and-beds-in-india/Hospitals and beds maintained by Railways.csv')


# In[ ]:


df2.head()


# In[ ]:


df2.tail()


# In[ ]:


# change column names
df2.rename (columns = {'Unnamed: 1': 'Zone / PU', 'Unnamed: 2':'Total No. of Hospitals', 'Unnamed: 3' : 'Total No. of Indoor Beds'}, inplace = True)
# drop first row
df2 = df2.drop([0,26])
df2 = df2.drop('Number of Hospitals and beds in Railways (as on 21/03/2018)', axis =1)


# In[ ]:


df2.info()


# In[ ]:


# change data type of number columns
df2[['Total No. of Hospitals', 'Total No. of Indoor Beds']] = df2[['Total No. of Hospitals', 'Total No. of Indoor Beds']].astype(int)


# In[ ]:


plt.figure(figsize=(8,8))
plt.barh(df2['Zone / PU'],df2['Total No. of Hospitals'], label = 'No. of hospital maintained by Railways')
plt.ylabel('Zone / PU')
plt.xlabel('No. of hospitals')
plt.legend()
plt.show()


# In[ ]:


plt.figure(figsize=(8,8))
plt.barh(df2['Zone / PU'],df2['Total No. of Indoor Beds'], label = 'No. of beds')
plt.ylabel('Zone / PU')
plt.xlabel('No. of beds')
plt.legend()
plt.show()


# In[ ]:


## Hospitals and Beds maintained by Ministry of Defence
# read data
df3 = pd.read_csv('/kaggle/input/hospitals-and-beds-in-india/Hospitals and Beds maintained by Ministry of Defence.csv')


# In[ ]:


df3.head()


# In[ ]:


df3.tail()


# In[ ]:


df3 = df3.drop('S. No.', axis =1)
df3 = df3.drop([29,30])


# In[ ]:


plt.figure(figsize=(10,10))
plt.barh(df3['Name of State'],df3['No. of Hospitals'], label = 'No. of hospitalmaintain by Ministry of Defence')
plt.ylabel('Name of state')
plt.xlabel('No. of hospital')
plt.legend()
plt.show()


# In[ ]:


plt.figure(figsize=(10,10))
plt.barh(df3['Name of State'],df3['No. of beds'], label = 'No. of beds')
plt.ylabel('Name of state')
plt.xlabel('No. of beds')
plt.legend()
plt.show()

