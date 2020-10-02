#!/usr/bin/env python
# coding: utf-8

# ![CDC](https://www.cdc.gov/coronavirus/2019-ncov/images/2019-coronavirus.png)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/ecdc-covid-data/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ### We will use additional data from [ECDC](https://www.ecdc.europa.eu/en/publications-data/download-todays-data-geographic-distribution-covid-19-cases-worldwide) (Thanks to [panos](https://www.kaggle.com/panosc))
# 

# In[ ]:


df = pd.read_excel('/kaggle/input/ecdc-covid-data/COVID-19-geographic-disbtribution-worldwide-2020-03-17.xlsx')


# Take an example,

# In[ ]:


df[df['Countries and territories']=='Taiwan']


# If we look at the data, seems that the cases weren't cumulative data. 

# In[ ]:


df.dtypes


# # Countries Affected

# In[ ]:


country = df['Countries and territories'].unique()
print(country)

print("Number of countries Infected: ",len(country))


# In[ ]:


df.head()


# Take the Abbreviation 

# In[ ]:


abbr = df.iloc[:,6:].drop_duplicates()


# In[ ]:


len(df['Countries and territories'].value_counts())


# In[ ]:


case = pd.DataFrame(df.groupby('Countries and territories')['Cases'].sum())
case['Countries and territories'] = case.index
case.index = np.arange(1,146,1)

worldwide = case[['Countries and territories','Cases']]


# ## Sorted by Cases

# In[ ]:


worldwide.sort_values(['Cases'], ascending=False)


# In[ ]:


worldwide = pd.merge(abbr,worldwide,on='Countries and territories')


# In[ ]:


worldwide


# We will add the coordinates from https://www.kaggle.com/eidanch/counties-geographic-coordinates

# In[ ]:


countries = pd.read_csv("../input/counties-geographic-coordinates/countries.csv")


# In[ ]:


countries


# In[ ]:


world = pd.merge(worldwide,countries,how='left',left_on='GeoId',right_on='country')
world 


# In[ ]:


world = world.drop(['name','country'], axis=1)


# In[ ]:


world.isna().sum()


# For a while, we will drop 

# In[ ]:


plot = world.dropna()


# In[ ]:


plot.head()


# In[ ]:


import folium 

map_dist = folium.Map()

for latitude, longitude, case, country in zip(plot['latitude'], plot['longitude'], plot['Cases'], plot['Countries and territories']):
    folium.CircleMarker([latitude, longitude], radius=case*0.0005,  popup = (str(country) +'\n\n' +'Cases: ' + str(case)), color='red', fill_color='red').add_to(map_dist)
map_dist


# In[ ]:


worldwide.sort_values('Cases',ascending=False)


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

plt.figure(figsize=(15,10))
ax = sns.barplot(x="Countries and territories", y="Cases", data=worldwide.sort_values('Cases',ascending=False)[:10])
plt.title('Top 10 Countries with Covid-19 Cases')


# ## TBA: Time series cases

# In[ ]:




