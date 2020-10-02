#!/usr/bin/env python
# coding: utf-8

#  <font size="3" color ="#990018"  >Hi ,Today we will conduct data analysis using the earthquakes in Turkey data we have. </font>
# * What we want to know:
# 1. Correlation between features
# 1. Which year had the most earthquakes in Turkey?
# 1. Where was the most earthquake?
# 1. Which country was the most earthquake?
# 1. How long did the earthquake last?
# 1. Where and when did the most severe earthquake occur?
# 
# 
# 
#  So let's start.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns #visualization
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("../input/earthquake/earthquake.csv")


# In[ ]:


data.head(10)


# In[ ]:


data.info


# In[ ]:


data.columns


# 1. lat - latitude of earthquake
# 1. dist - distance of direction in km
# 1. richter - intensity of earthquake (Richter)
# 1. md - depending on time magnitude
# 1. mw - moment magnitude 
# 1. ms - surface-wave magnitude
# 1. mb - body-wave magnitude
# * **xm** - biggest magnitude value in specified magnitude values

# 
# <font size="3" color ="#990018" >Firstly, we must create a new column only years
# </font>

# In[ ]:


def yeardate(x):
    return x[0:4]
data["yeardate"] = data.date.apply(yeardate)
#We must change object to integer.
data['yeardate'] = data.yeardate.astype(int)
print(data.yeardate.dtypes)
data.head(3)


# <font size="3" color ="#990018">Correlation between features
# </font>

# In[ ]:


plt.figure(figsize=(15,15))
sns.heatmap(data.corr(), annot = True, fmt= ".1f", linewidths = .3)
plt.show()


# We have possitive correlation **xm** between as **mw**, **ms** and **mb**
# 

# <font size="3" color ="#990018" >Which year had the most earthquakes in Turkey?</font>
# 

# In[ ]:


data.yeardate.plot(kind = "hist" , color = "red" , edgecolor="black", bins = 100 , figsize = (12,12) , label = "Earthquakes frequency")
plt.legend(loc = "upper right")
plt.xlabel("Years")
plt.show()


# Maybe it is just a lost data but most earthquake occurred in 2000-2017
# <font size="3" color="990018">Where was the most earthquake?</font>
# 

# In[ ]:


data.city.value_counts().plot(kind = "bar" , color = "red" , figsize = (30,10),fontsize = 20)
plt.xlabel("City",fontsize=18,color="blue")
plt.ylabel("Frequency",fontsize=18,color="blue")
plt.show()


#  <font size="3" color="990018">Which country-area was the most earthquake?</font>
# 

# In[ ]:


data.country.value_counts().plot(kind = "bar" , color = "red" , figsize = (30,10),fontsize = 20)
plt.xlabel("Country",fontsize=18,color="blue")
plt.ylabel("Frequency",fontsize=18,color="blue")
plt.show()


# <font size="3" color="990018">How long did the earthquake last?</font>
# 

# In[ ]:


data.long.max()
filtre = data.long == 48.0
data[filtre]


#  <font size="3" color="990018">Where and when did the most severe earthquake occur?</font>

# In[ ]:


data.xm.max()
filtering = data.country == "turkey"
filtering2 = data.xm == 7.9
data[filtering & filtering2]


#  <font size="3" color="990018">Earthquake - Magnitude level</font>

# In[ ]:


threshold = sum(data.xm) / len(data.xm)
data["magnitude-level"] = ["hight" if i > threshold else "low" for i in data.xm]
data.loc[:10,["magnitude-level","xm","city"]]

