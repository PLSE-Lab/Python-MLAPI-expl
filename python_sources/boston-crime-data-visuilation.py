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


import numpy as np


# In[ ]:


import pandas as pd


# In[ ]:


import os


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


import seaborn as sns


# In[ ]:


path = "../input"


# In[ ]:


os.chdir(path)


# In[ ]:


data = pd.read_csv("../input/crime.csv", encoding = "ISO-8859-1", low_memory=False)


# In[ ]:


data.head()


# In[ ]:


data.tail()


# In[ ]:


data.columns


# In[ ]:


data.DISTRICT.value_counts()


# In[ ]:


fig = plt.figure(figsize = (5,5))


# In[ ]:


ax = fig.add_subplot(111)


# In[ ]:


sns.countplot("DISTRICT", data = data, ax = ax)


# In[ ]:


plt.show()


# In[ ]:


plt.figure(figsize=(16,8))


# In[ ]:


data['DISTRICT'].value_counts().plot.bar()


# In[ ]:


sns.catplot(x="DISTRICT",       # Variable whose distribution (count) is of interest
            hue="MONTH",      # Show distribution, pos or -ve split-wise
            col="YEAR",       # Create two-charts/facets, gender-wise
            data=data,
            kind="count")


# In[ ]:


plt.figure(figsize=(16,8))


# In[ ]:


data['DISTRICT'].loc[data['YEAR']==2015].value_counts().plot.bar()


# In[ ]:


plt.figure(figsize=(16,8))


# In[ ]:


data['OFFENSE_CODE_GROUP'].value_counts().plot.bar()


# In[ ]:


plt.show()


# In[ ]:


plt.figure(figsize=(16,8))


# In[ ]:


sns.countplot(x='YEAR', data = data)


# In[ ]:


plt.figure(figsize=(16,8))


# In[ ]:


top10cloc = data.groupby('DISTRICT')['INCIDENT_NUMBER'].count().sort_values(ascending=False)


# In[ ]:


top10cloc = top10cloc [:10]


# In[ ]:


top10cloc.plot(kind='bar', color='green')


# In[ ]:


bot10cloc = data.groupby('DISTRICT')['INCIDENT_NUMBER'].count().sort_values(ascending=True)


# In[ ]:


bot10cloc = bot10cloc [:10]


# In[ ]:


bot10cloc.plot(kind='bar', color='blue')


# In[ ]:


plt.figure(figsize=(15,7))


# In[ ]:


data.groupby(['DAY_OF_WEEK'])['INCIDENT_NUMBER'].count().plot(kind = 'bar')


# In[ ]:


plt.figure(figsize=(17,9))


# In[ ]:


data.groupby(['DISTRICT'])['STREET'].count().plot(kind = 'bar')


# In[ ]:


groups = data['DISTRICT'].unique()


# In[ ]:


n_groups = len(data['DISTRICT'].unique())-1


# In[ ]:


index = np.arange(n_groups)


# In[ ]:


bar_width = 0.2


# In[ ]:


opacity= 0.8


# In[ ]:


plt.figure(figsize=(16,8))


# In[ ]:


dy = data[['DISTRICT','YEAR']]

dy_2015 = dy.loc[(dy['YEAR'] == 2015)]
dy_2016 = dy.loc[(dy['YEAR'] == 2016)]
dy_2017 = dy.loc[(dy['YEAR'] == 2017)]
dy_2018 = dy.loc[(dy['YEAR'] == 2018)]

cri_2015 = dy_2015['DISTRICT'].value_counts()
cri_2016 = dy_2016['DISTRICT'].value_counts()
cri_2017 = dy_2017['DISTRICT'].value_counts()
cri_2018 = dy_2018['DISTRICT'].value_counts()

bar1 = plt.bar(index, cri_2015, bar_width, alpha = opacity, color = 'r', label = '2015')
bar2 = plt.bar(index + bar_width, cri_2016, bar_width, alpha = opacity, color = 'g', label = '2016')
bar3 = plt.bar(index+ bar_width+ bar_width, cri_2017, bar_width, alpha = opacity, color = 'b', label = '2017')
bar4 = plt.bar(index+ bar_width+ bar_width+ bar_width, cri_2018, bar_width, alpha = opacity, color = 'y', label = '2018')


# In[ ]:


from mpl_toolkits.basemap import Basemap, cm


# In[ ]:


import descartes


# In[ ]:


import geopandas as gpd


# In[ ]:


from shapely.geometry import Point, Polygon


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


fig,ax = plt.subplots(figsize = (15,15))


# In[ ]:


geometry = [Point(xy) for xy in zip( data["Long"],data["Lat"])]


# In[ ]:


geometry[:3]


# In[ ]:


geo_data = gpd.GeoDataFrame(data, geometry = geometry)


# In[ ]:


geo_data.head()


# In[ ]:


fig,ax = plt.subplots(figsize = (15,15))


# In[ ]:


from mpl_toolkits.basemap import Basemap


# In[ ]:


import folium


# In[ ]:


from folium import plugins


# In[ ]:


import statsmodels.api as sm


# In[ ]:


data[['Lat','Long']].describe()


# In[ ]:


location = data[['Lat','Long']]


# In[ ]:


location = location.dropna()


# In[ ]:


location = location.loc[(location['Lat']>40) & (location['Long'] < -60)]


# In[ ]:


x = location['Long']


# In[ ]:


y = location['Lat']


# In[ ]:


colors = np.random.rand(len(x))


# In[ ]:


plt.figure(figsize=(20,20))


# In[ ]:


plt.scatter(x, y,c=colors, alpha=0.5)


# In[ ]:


plt.show()


# In[ ]:


m = folium.Map([42.348624, -71.062492], zoom_start=11)


# In[ ]:


m


# In[ ]:


x = location['Long']


# In[ ]:


y = location['Lat']


# In[ ]:


sns.jointplot(x, y, kind='scatter')


# In[ ]:


sns.jointplot(x, y, kind='hex')


# In[ ]:


sns.jointplot(x, y, kind='kde')

