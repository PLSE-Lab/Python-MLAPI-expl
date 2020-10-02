#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.plotly  as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected = True)
import plotly.graph_objs as go
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("../input/crime.csv",encoding='ISO-8859-1')


# In[ ]:


data.info()


# In[ ]:


columns = []
for each in data.columns:
    columns.append (each.lower())
data.columns = columns
data.sample()


# In[ ]:


data.shooting.value_counts()


# In[ ]:


sns.countplot (x =data.shooting, hue="district" , data =data)


# In[ ]:


sns.countplot (x="shooting", hue="year" , data =data ,palette = "Set3")


# In[ ]:


#sns.countplot(x=data.shooting, hue =data.day_of_week)
data.day_of_week.value_counts().plot.bar()


# In[ ]:


sns.catplot(x="shooting",       # Variable whose distribution (count) is of interest
            hue="district",      # Show distribution, pos or -ve split-wise
            col="year",       # Create two-charts/facets, gender-wise
            data=data,
            kind="count"
            )


# In[ ]:


plt.figure(figsize=(16,8))
data.district.value_counts().plot.bar()
plt.grid()


# In[ ]:


data.district.loc[data.year ==2015].value_counts().plot.bar() 
plt.show()
data.district.loc[data.year ==2016].value_counts().plot.bar() 
plt.show()
data.district.loc[data.year ==2017].value_counts().plot.bar() 
plt.show()
data.district.loc[data.year ==2018].value_counts().plot.bar() 
plt.show()


# In[ ]:


sns.countplot(data.offense_code_group[:5],hue=data.year)


# In[ ]:


data.hour.value_counts()


# In[ ]:


data.hour.value_counts().plot.bar() 
plt.show()


# In[ ]:


plt.figure(figsize=(20,8))
data.offense_code_group.value_counts().plot.bar()


# In[ ]:


from mpl_toolkits.basemap import Basemap, cm 
import descartes
import geopandas as gpd
from shapely.geometry import Point, Polygon
geometry = [Point(xy) for xy in zip( data["long"],data["lat"])]
geometry[:3]


# In[ ]:


geo_data = gpd.GeoDataFrame(data,geometry = geometry)
geo_data.head()


# In[ ]:


from mpl_toolkits.basemap import Basemap
import folium
from folium import plugins
import statsmodels.api as sm


# In[ ]:


data[['lat','long']].describe()


# In[ ]:


location = data[['lat','long']]
location=location.dropna()


# In[ ]:


location = location.loc[(location.lat>40) & (location.long<-60)]
x=location.long
y=location.lat
colors = np.random.rand(len(x))
plt.figure(figsize = (10,10))
plt.scatter(x,y,c=colors,alpha=0.5)


# In[ ]:


m = folium.Map([42.34,-71.06], zoom_start=11)
m


# In[ ]:


sns.jointplot(x, y, kind='hex')


# In[ ]:


# 5.6 When do serious crimes occur?
#We can consider patterns across several different time scales: hours of the day, days of the week, and months of the year.
# 5.6.1 Number of Crimes reported at Hour during the Day
plt.figure(figsize=(16,8))
sns.countplot(x='hour', data = data)
plt.ylabel('Number of Crimes')
plt.title('BOSTON: Hour wise # of Crimes Reported')
#plt.savefig("C:\\Users\\bk42969\\Desktop\\BigData\\CrimeHourDuringTheDay.png")
plt.show()
# Crimes are observed Least in the Early Hours of the Morning. 


# In[ ]:


datarob = data[data.offense_code_group=="Larceny"]
datarob


# In[ ]:


# 5.6 When do serious crimes occur?
#We can consider patterns across several different time scales: hours of the day, days of the week, and months of the year.
# 5.6.1 Number of Crimes reported at Hour during the Day

plt.figure(figsize=(16,8))
sns.countplot(x='hour', data = datarob)
plt.ylabel('Number of Crimes')
plt.title('BOSTON: Hour wise # of Crimes Reported')
#plt.savefig("C:\\Users\\bk42969\\Desktop\\BigData\\CrimeHourDuringTheDay.png")
plt.show()
# Crimes are observed Least in the Early Hours of the Morning. 


# In[ ]:


databallistics = data[data.offense_code_group=="Ballistics"]
plt.figure(figsize=(16,8))
sns.countplot(x='hour', data = databallistics)
plt.ylabel('Number of Crimes')
plt.title('BOSTON: Hour wise # of Crimes Reported')
#plt.savefig("C:\\Users\\bk42969\\Desktop\\BigData\\CrimeHourDuringTheDay.png")
plt.show()
# Crimes are observed Least in the Early Hours of the Morning. 


# **References**
# 
# 1. Crimes in Boston [https://www.kaggle.com/ankkur13/boston-crime-data/kernels](http://)
