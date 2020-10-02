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

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# Any results you write to the current directory are saved as output.


# **Let's look at Data :**

# In[ ]:


# Read DATA
DATA  = pd.read_csv('../input/earthquake/1398-08-16 1.12.03 AM.csv')

# Clear columns we do not need
DATA = DATA.drop(columns=['dis1','dis2','dis3','magnitude_type','mm','yy','local_time','day_name','dd'])

# Rename Columns
DATA = DATA.rename(columns={'n':'North','e':'East','ew':'East west','ns':'North South','gmt_date':'date','gmt_time':'time'})

# Change the Data Type
DATA['depth'] = DATA['depth'].apply(lambda x: int(x))
DATA['East'] = DATA['East'].apply(lambda x:float(x))
DATA['East west'] = DATA['East west'].apply(lambda x:float(x))
DATA['magnitude'] = DATA['magnitude'].apply(lambda x:float(x))
DATA['North'] = DATA['North'].apply(lambda x:float(x))
DATA['North South'] = DATA['North South'].apply(lambda x:float(x))


# Drop NAN values 
DATA = DATA.dropna()


DATA.head()


# In[ ]:


# converting Date column to date
DATA.date = pd.to_datetime(DATA.date,format = '%Y-%m-%d')

# Make Weekday
DATA['Weekday'] = DATA.date.apply(lambda x: x.dayofweek)

# Extract the Month
DATA['Month'] = DATA.date.apply(lambda x:x.month)

# Extract the Day
DATA['Day'] = DATA.date.apply(lambda x: x.day)

# Extract The Year
DATA['Year'] = DATA.date.apply(lambda x:x.year)

# Converting Time column To time(H:M:S)
DATA.time = pd.to_datetime(DATA['time'])
# Extracting the Hour
DATA['Hour'] = DATA.time.apply(lambda x:x.hour)

# Extracting The minute
DATA['Minute'] = DATA.time.apply(lambda x:x.minute)

# Show 5 random sample

DATA.sample(5)


# In[ ]:


# Coreelation between features

plt.figure(figsize = (15,15))
sns.heatmap(DATA.corr(),annot = True , fmt = '.1f',linewidths= .3)
plt.show()


# In[ ]:


# Which year had the most earthquakes in IR

DATA.Year.plot(kind = 'hist',color = 'red',edgecolor = 'black',bins = 100 , figsize = (12,12),label = 'Earthquakes frequency')
plt.legend(loc = 'upprt right')
plt.xlabel('years')
plt.show()


# In[ ]:


DATA.location.value_counts().plot(kind = 'bar',color = 'red',figsize =(30,10),fontsize = 20)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['B Nazanin', 'Tahoma']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel('City',fontsize = 18, color = 'blue')
plt.ylabel('frequency',fontsize = 18,color = 'blue')
plt.show()


# In[ ]:


DATA.depth.max()
filtre = DATA.depth == 50.0
DATA[filtre]


# In[ ]:


DATA[DATA['magnitude'] >= 7.4]


# In[ ]:


threshold = sum(DATA.magnitude) / len(DATA.magnitude)
DATA['magnitude-level'] = ['height' if i > threshold else 'low' for i in DATA.magnitude]

DATA.loc[:10,['magnitude-level','magnitude','location']]


# In[ ]:


dataover5 = DATA[DATA.magnitude >= 5]

plt.figure(figsize = (15,5))
plt.subplot(1,2,1)
dataover5.Month.value_counts().sort_index().plot.bar()
plt.xlabel('Months of The year')
plt.subplot(1,2,2)
dataover5.Weekday.value_counts().sort_index().plot.bar()
plt.xlabel('Days of the Week')


# In[ ]:


plt.figure(figsize = (15,5))
plt.subplot(1,2,1)
plt.plot(DATA.Year.value_counts().sort_index())
plt.subplot(1,2,2)
plt.plot(DATA[DATA.magnitude >= 5].Year.value_counts().sort_index())


# In[ ]:


DATA.Year.value_counts().sort_index(ascending = False).plot.area()
DATA[DATA.magnitude >= 4].Year.value_counts().sort_index(ascending = False).plot.area()
plt.legend(['all earthquakes','5 plus'])

