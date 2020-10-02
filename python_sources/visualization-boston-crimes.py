#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


import numpy as np


# In[4]:


import pandas as pd


# In[5]:


import os


# In[6]:


import matplotlib.pyplot as plt


# In[7]:


import seaborn as sns


# In[8]:


path = "../input"


# In[9]:


os.chdir(path)


# In[10]:


data = pd.read_csv("../input/crime.csv", encoding = "ISO-8859-1", low_memory=False)


# In[11]:


data.head()


# In[12]:


data.tail()


# In[13]:


data.columns


# In[14]:


data.DISTRICT.value_counts()


# In[15]:


fig = plt.figure(figsize = (5,5))


# In[16]:


ax = fig.add_subplot(111)


# In[17]:


sns.countplot("DISTRICT", data = data, ax = ax)


# In[19]:


plt.show()


# In[20]:


plt.figure(figsize=(16,8))


# In[21]:


data['DISTRICT'].value_counts().plot.bar()


# In[22]:


sns.catplot(x="DISTRICT",       # Variable whose distribution (count) is of interest
            hue="MONTH",      # Show distribution, pos or -ve split-wise
            col="YEAR",       # Create two-charts/facets, gender-wise
            data=data,
            kind="count")


# In[23]:


plt.figure(figsize=(16,8))


# In[24]:


data['DISTRICT'].loc[data['YEAR']==2015].value_counts().plot.bar()


# In[25]:


plt.figure(figsize=(16,8))


# In[26]:


data['OFFENSE_CODE_GROUP'].value_counts().plot.bar()


# In[31]:


plt.show()


# In[30]:


plt.figure(figsize=(16,8))


# In[32]:


sns.countplot(x='YEAR', data = data)


# In[33]:


plt.figure(figsize=(16,8))


# In[35]:


top10cloc = data.groupby('DISTRICT')['INCIDENT_NUMBER'].count().sort_values(ascending=False)


# In[37]:


top10cloc = top10cloc [:10]


# In[38]:


top10cloc.plot(kind='bar', color='green')


# In[39]:


bot10cloc = data.groupby('DISTRICT')['INCIDENT_NUMBER'].count().sort_values(ascending=True)


# In[40]:


bot10cloc = bot10cloc [:10]


# In[41]:


bot10cloc.plot(kind='bar', color='blue')


# In[42]:


plt.figure(figsize=(15,7))


# In[43]:


data.groupby(['DAY_OF_WEEK'])['INCIDENT_NUMBER'].count().plot(kind = 'bar')


# In[44]:


plt.figure(figsize=(17,9))


# In[45]:


data.groupby(['DISTRICT'])['STREET'].count().plot(kind = 'bar')


# In[47]:


groups = data['DISTRICT'].unique()


# In[48]:


n_groups = len(data['DISTRICT'].unique())-1


# In[49]:


index = np.arange(n_groups)


# In[50]:


bar_width = 0.2


# In[51]:


opacity= 0.8


# In[52]:


plt.figure(figsize=(16,8))


# In[53]:


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


# In[54]:


from mpl_toolkits.basemap import Basemap, cm


# In[55]:


import descartes


# In[56]:


import geopandas as gpd


# In[57]:


from shapely.geometry import Point, Polygon


# In[58]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[59]:


fig,ax = plt.subplots(figsize = (15,15))


# In[60]:


geometry = [Point(xy) for xy in zip( data["Long"],data["Lat"])]


# In[61]:


geometry[:3]


# In[62]:


geo_data = gpd.GeoDataFrame(data, geometry = geometry)


# In[63]:


geo_data.head()


# In[64]:


fig,ax = plt.subplots(figsize = (15,15))


# In[65]:


from mpl_toolkits.basemap import Basemap


# In[66]:


import folium


# In[67]:


from folium import plugins


# In[68]:


import statsmodels.api as sm


# In[69]:


data[['Lat','Long']].describe()


# In[70]:


location = data[['Lat','Long']]


# In[71]:


location = location.dropna()


# In[72]:


location = location.loc[(location['Lat']>40) & (location['Long'] < -60)]


# In[73]:


x = location['Long']


# In[74]:


y = location['Lat']


# In[75]:


colors = np.random.rand(len(x))


# In[76]:


plt.figure(figsize=(20,20))


# In[77]:


plt.scatter(x, y,c=colors, alpha=0.5)


# In[78]:


plt.show()


# In[79]:


m = folium.Map([42.348624, -71.062492], zoom_start=11)


# In[80]:


m


# In[81]:


x = location['Long']


# In[82]:


y = location['Lat']


# In[83]:


sns.jointplot(x, y, kind='scatter')


# In[84]:


sns.jointplot(x, y, kind='hex')


# In[85]:


sns.jointplot(x, y, kind='kde')

