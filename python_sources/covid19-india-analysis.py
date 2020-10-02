#!/usr/bin/env python
# coding: utf-8

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


import seaborn as sns
import matplotlib.pyplot as plt


# **Loading of data**

# In[ ]:


df=pd.read_csv("../input/covid19-corona-virus-india-dataset/complete.csv")


# In[ ]:


df.head()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.columns

Renamed data
# In[ ]:


df.rename(columns={'Name of State / UT':'State','Total Confirmed cases (Indian National)':'Confirmed_cases_india', 'Total Confirmed cases (Foreign National)':' Confirmed_cases_Foreign'
                 ,'Total Confirmed cases *': 'Total_Confirmed'},inplace=True)


# In[ ]:


df.columns


# In[ ]:


df.head()


# In[ ]:


df['Total_Confirmed'].plot.hist()


# In[ ]:


sns.countplot(x='Death',data=df,hue='State',palette='mako')


# confirmed cases
# 

# In[ ]:


df['Total_Confirmed'].describe()


# **Top state**

# In[ ]:


top=df.nlargest(20,'Total_Confirmed')


# In[ ]:


sns.stripplot(x='State',y='Total_Confirmed',data=top)


# In[ ]:


fig,ax = plt.subplots(1)
fig.set_size_inches(14,6)
sns.barplot(df["State"],df["Total_Confirmed"])
plt.xticks(rotation=45,fontsize=10)
plt.title("Total Confirmed Cases Statewise as on ",fontsize=16)
plt.xlabel("State/Union Territory",fontsize=14)
plt.ylabel("Total Confirmed Cases",fontsize=14)
plt.show()


# In[ ]:


from matplotlib import style
style.use('ggplot')

df.plot(x='Date',y='Total_Confirmed',kind='line',linewidth=5,color='R',figsize=(25,15))
plt.ylabel('Corona Cases')

plt.grid()
plt.show()


# **Regression plot**

# In[ ]:


sns.lmplot(x='Total_Confirmed',y='Death',data=df,hue='State',size=8)


# In[ ]:


Cured/Discharged/Migrated	Latitude	Longitude	Death	Total_Confirmed


# In[ ]:



from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame

get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')

geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
geo_df = GeoDataFrame(df, geometry=geometry)
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
geo_df.plot(ax=world.plot(figsize=(15, 15)), marker='o', color='red', markersize=15);


# In[ ]:




