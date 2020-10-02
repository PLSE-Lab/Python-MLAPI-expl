#!/usr/bin/env python
# coding: utf-8

# In[27]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from mpl_toolkits.basemap import Basemap #plotting
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

import warnings
warnings.simplefilter("ignore") # drop warnings
pd.options.display.max_columns = 200 # for showing all columns of tables



# In[2]:


data = pd.read_csv("../input/globalterrorismdb_0616dist.csv", encoding='ISO-8859-1').set_index("eventid")


# In[12]:


data['date'] = data.index.map(lambda index: '.'.join([str(index)[6:8],str(index)[4:6],str(index)[0:4]]))
data.tail()


# In[17]:


russia_data = data.loc[data.country_txt == 'Russia',:]           


# In[22]:


russia_data.latitude[russia_data.longitude.isnull()]


# In[23]:


russia_data = russia_data.loc[:,('city','latitude','longitude', 'suicide', 'attacktype1_txt', 'success', 'nkill','date')]


# In[26]:


russia_data = russia_data[russia_data.latitude.notnull()]


# In[30]:


def draw_map(df, z=1):
    zoom = (10/3) + (1/3) * z
    m =  Basemap(projection='merc', llcrnrlon=df.longitude.min()-z,llcrnrlat=df.latitude.min()-z,urcrnrlon=df.longitude.max()+z,urcrnrlat=df.latitude.max()+z)
    m.bluemarble()
    x,y = df[list(df.longitude), list(df.latitude)]
    m.scatter()
    plt.show()


# In[32]:


draw_map(russia_data, 10)

