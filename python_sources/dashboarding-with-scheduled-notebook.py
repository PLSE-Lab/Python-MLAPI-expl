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


# In[ ]:


data = pd.read_csv("../input/fire-department-calls-for-service.csv")

data.head(10)


# In[ ]:


colWithMixedType = (1,9,20,25,29,30)

data.dtypes


# In[ ]:


Latitude = []
Longitude = []

for i in range(len(data['Location'])):
    x = data['Location'].iloc[i].split('\'')
    Latitude.append(float(x[5]))
    Longitude.append(float(x[9]))
    
data['Latitude'] = pd.Series(Latitude)
data['Longitude'] = pd.Series(Longitude)


# # Missing Values

# In[ ]:


print(data.shape[0])

per = (data.isnull().sum()/data.shape[0])*100
percents = per.iloc[per.nonzero()[0]]

print(percents)

from matplotlib import pyplot as plt
percents.plot.barh()
plt.show()


# Removing rows where percentage is less than 2.

# In[ ]:


data = data[data['Available DtTm'].notnull()]
data = data[data['City'].notnull()]
data = data[data['Zipcode of Incident'].notnull()]
data = data[data['Station Area'].notnull()]
data = data[data['Box'].notnull()]
data = data[data['Original Priority'].notnull()]
data = data[data['Priority'].notnull()]
data = data[data['Unit sequence in call dispatch'].notnull()]


# In[ ]:


print(data.shape[0])

per = (data.isnull().sum()/data.shape[0])*100
percents = per.iloc[per.nonzero()[0]]

print(percents)


# # Looking at each variable one at a time

# In[ ]:


len(data['Call Number'].unique())


# In[ ]:


len(data['Unit ID'].unique())


# In[ ]:


len(data['Call Type'].unique())


# In[ ]:


len(data['Call Date'].unique())


# In[ ]:





# # Plots
# 

# In[ ]:



data["Call Number"].value_counts().nlargest(30).plot("barh",width = 1).invert_yaxis()
plt.title("Call Number")
plt.xlabel('Count')
plt.ylabel('Call Number')
plt.show()

data["Unit ID"].value_counts().nlargest(30).plot("barh",width = 1).invert_yaxis()
plt.title("Unit ID")
plt.xlabel('Count')
plt.ylabel('Unit ID')
plt.show()

data["Call Type"].value_counts().nlargest(20).plot("barh",width = 1).invert_yaxis()
plt.title("Call Type")
plt.xlabel('Count')
plt.ylabel('Call Type')
plt.show()

data["Call Date"].value_counts().nlargest(30).plot("barh",width = 1).invert_yaxis()
plt.title("Call Date")
plt.xlabel('Count')
plt.ylabel('Call Date')
plt.show()

data["Zipcode of Incident"].value_counts().nlargest(30).plot("barh",width = 1).invert_yaxis()
plt.title("Zipcode of Incident")
plt.xlabel('Count')
plt.ylabel('Zipcode of Incident')
plt.show()


# In[ ]:


from mpl_toolkits.basemap import Basemap
import folium
from folium import plugins

mm = folium.Map([37.7749, -122.4194], tiles = "Stamen Terrain", zoom_start=12)

hm_wide = plugins.HeatMap( list(zip(data.Latitude.values, data.Longitude.values)),
                     min_opacity=0.2,
                     radius=17, blur=15,
                     max_zoom=1
                 )
mm.add_child(hm_wide)

mm


# In[ ]:




