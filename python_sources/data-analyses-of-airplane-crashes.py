#!/usr/bin/env python
# coding: utf-8

# **Data Analyses of Airplane Crashes**

# In last 100 years aircraft technology improved tremendously but this improvements were results of the aircraft crashes and behind this improvements so many people lost their lives. In this kernel we will analyse aircraft crashes and see whether or not some features have effect on it.
# 

# SUMMARY
# 
# * [ CONVERTING DATES TO DATETIME](#1)
# * [CLEANING DATA.ABOARD AND DATA.FATALITIES COLUMNS](#2)
# * [YEARLY SUM OF FATALITIES AND ABOARD](#3)
# * [BUILDING DATA.TIME COLUMN BY USING REGULAR EXPRESSION](#4)
# * [EXTRACTING AIRCRAFT BRANDS FROM DATA.AC_TYPE COLUMN ](#5)
# * [CLASSIFYING DEATH RATES](#6)
# * [TOP 7 DEADLIEST AIRCRAFTS IN HISTORY](#7)
# * [CLEANING DATA.TIME TO MAKE A REALISTIC PLOT](#8)
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import Counter
# plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import seaborn as sns
# word cloud library
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import datetime as dt

# matplotlib

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/planecrashinfo_20181121001952.csv')
data.info()


# **In this dataset we only use ['date','time','ac_type','aboard','fatalities '] columns**

# <a id="1"></a> 
# ##  CONVERTING DATES TO DATETIME

# In[ ]:


data.head()


# In[ ]:


date = []
for each in data.date:
    x = pd.to_datetime(each)
    date.append(x)
    
data.date = date


# <a id="2"></a> 
# ## CLEANING DATA.ABOARD AND DATA.FATALITIES COLUMNS

# In[ ]:


seperate = data.aboard.str.split()
a,b,c = zip(*seperate)
seperate = data.fatalities.str.split()
d,e,f = zip(*seperate)
data.aboard = a
data.fatalities = d


# In[ ]:


#data.dropna(inplace = True)
data.aboard.replace(['?'],0,inplace = True)
data.ac_type.replace(['?'],'a b',inplace = True)
data.fatalities.replace(['?'],0,inplace = True)


# In[ ]:


data.aboard = data.aboard.astype(float)
data.fatalities = data.fatalities.astype(float)


# In[ ]:


data.head()


# <a id="3"></a> 
# ## YEARLY SUM OF FATALITIES AND ABOARD

# In[ ]:


sum_fatalities = []
sum_aboard = []
year = data.date.dt.year.unique()
for each in year:
    x = sum(data.fatalities[data.date.dt.year==each])
    y = sum(data.aboard[data.date.dt.year==each])
    sum_fatalities.append(x)
    sum_aboard.append(y)
    


# In[ ]:


trace1 = go.Bar(
    x = year,
    y = sum_fatalities,
    marker = dict(color = 'blue'),
    name = 'death :'
    )
trace2 = go.Scatter(
    x = year,
    y = sum_aboard,
    mode = 'markers+lines',
    marker = dict(color = 'red'),
    name = 'aboard :'
    )
layout = dict(title = 'DEATHS AND ABOARD',
              xaxis= dict(title= 'YEAR',ticklen= 15,zeroline= False),yaxis = dict(title = 'NUMBERS',ticklen= 15,zeroline= False))
data1 = [trace1,trace2]
fig = dict(data=data1,layout=layout)
iplot(fig)


# So between 60s and 80s were the deadliest time for aircraft history. Very first period of large body and turbojet engined aircrafts like de havillands comet series ,dougles dc series and boeing 707s.

# 
# <a id="4"></a> 
# ## BUILDING DATA.TIME COLUMN BY USING REGULAR EXPRESSION

# In[ ]:


data['time'].replace(['?'],'00:00',inplace = True)
import re
time = []
for each in data.time:
    x = re.sub('[^0-9]','',each)
    x = re.sub(' ','',x)
    if len(x)!=4:
        x = '0000'
    a = list(x)
    a.insert(2,':')
    a = ''.join(a)
    time.append(a)
   
data['time'] = time


# 
# <a id="5"></a> 
# ## EXTRACTING AIRCRAFT BRANDS FROM DATA.AC_TYPE COLUMN

# In[ ]:


sep = data.ac_type.str.split() 
brand = []
for each in sep:
  
    brand.append( each[0])
data.ac_type = brand


# 
# <a id="6"></a> 
# ## CLASSIFYING DEATH RATES

# In[ ]:


data['death_rate'] = data.fatalities/data.aboard


data['dead_or_alive'] = ['alive' if each<0.5 else 'dead' for each in data.death_rate]

data.head()


# I want to classify aboards and fatalities together by dividing them each other in order to use them in swarmplot.

# 
# <a id="7"></a> 
# ## TOP 7 DEADLIEST AIRCRAFTS IN HISTORY
# 

# In[ ]:


x = Counter(data.ac_type)

y = x.most_common(7)

a,b = zip(*y)

trace1 = go.Pie(
    values = b,
    labels = a)

data2 = [trace1]

layout = dict(title = 'TOP 7 DEADLIEST AIRCRAFTS IN HISTORY')

fig = dict(data = data2,layout = layout)

iplot(fig)


# In[ ]:


a = Counter(data.ac_type)

b = a.most_common(5)

x,y = zip(*b)

a = data[data.ac_type==x[0]]

b = data[data.ac_type==x[1]]

c = data[data.ac_type==x[2]]

d = data[data.ac_type==x[3]]

e = data[data.ac_type==x[4]]

data1 = pd.concat([a,b,c,d,e],axis=0)


# In[ ]:


plt.figure(figsize=(10,10))

sns.swarmplot(data1.ac_type,data1.date.dt.year,hue=data.dead_or_alive)


# In[ ]:


len(data)


# 
# 
# <a id="8"></a> 
# ## CLEANING DATA.TIME TO MAKE A REALISTIC PLOT
# 

# In[ ]:


a = data[data.time == '00:00'].index
data.drop(a,inplace = True)


# In[ ]:


a = pd.to_datetime(data.time)

b = a.dt.hour.values

data['time'] = b

len(data)


# In[ ]:


data.head()


# In[ ]:


trace1 = go.Histogram(
    x = data.time,
    opacity = 0.5,
    marker = dict(color = 'blue'))
layout = dict(title = 'AC-CRASHES PER HOUR',xaxis = dict(title = 'HOURS'),yaxis = dict(title = 'CRASH NUMBER'))
dat = [trace1]
fig = {'data':dat,'layout':layout}
iplot(fig)


# In[ ]:


plt.figure(figsize=(10,10))

data.groupby(data.time).death_rate.mean().plot()
plt.title('time-death rate correlation')
plt.xlabel('time')
plt.ylabel('death rate')


# When we compare time-death rate correlation and AC-CRASHES PER HOUR graph we can observe that death rate and crash number
# is not directly related each other. Even though most crashes happen in daylight dark is deadlier than daylight.
