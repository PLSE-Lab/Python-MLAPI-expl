#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # visialization
import seaborn as sns 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../input/avocado.csv") #read file


# In[ ]:


df.info() #quick overview of the data


# * There is no null data in any column, this is good news.
# * Column names are not perfect.
# * There are gaps between words
# * Not possible to understand meaning intuitively
# 

# In[ ]:


df.columns = df.columns.str.replace(' ', '_')
df = df.drop('Unnamed:_0',1)
df.head()


# Lets make Date column datetime format

# In[ ]:


df.Date = pd.to_datetime(df.Date)
df.head()


# In[ ]:


df.info()


# Lets check correlatin matrix

# In[ ]:


sns.heatmap(df.corr(),vmax = 1, vmin = 0,annot = True)
plt.show()


# In[ ]:


df.describe()


# In[ ]:


print(df.type.nunique()) #number of different types
print(df.year.nunique()) #number of different year
print(df.region.nunique()) #number of different regions


# In[ ]:


df.type.value_counts().plot(kind = 'bar')
plt.show()
df.type.value_counts()


# In[ ]:


df.region.value_counts()


# Lets look at the average prices of organic and conventional avocado.

# In[ ]:


index = np.arange(2)
objects = ['organic','conventional']
plt.bar(index,[df[df.type == "organic"].AveragePrice.mean(), df[df.type == "conventional"].AveragePrice.mean()])
plt.xticks(index,objects)
plt.show()


# Average price of organic is higher than conventional

# Lets see the organic and conventional avocado prices in box plot

# In[ ]:


objects = ['organic','conventional']
plt.boxplot([df[df.type == "organic"].AveragePrice,df[df.type == "conventional"].AveragePrice])
plt.xticks([1,2],objects)
plt.show()


# Lets see the average prices box plot for different years

# In[ ]:


datayear = []
for i in df.year.unique():
    datayear.append(df[df.year == i].AveragePrice)
plt.boxplot(datayear)
plt.xticks(range(1,df.year.nunique()+1),df.year.unique())
plt.show()


# Average price of organic and conventional avocado for different years

# In[ ]:


datayearorganic = []
datayearconventional = []
for i in df.year.unique():
    datayearorganic.append(df[(df.year == i) & (df['type'] == 'organic')].AveragePrice.mean())
    datayearconventional.append(df[(df.year == i) & (df.type == 'conventional')].AveragePrice.mean())
bar_width = 0.35
plt.bar(np.arange(df.year.nunique()),datayearorganic,bar_width, label = 'organic')
plt.bar(np.arange(df.year.nunique())+bar_width,datayearconventional,bar_width, label = 'conventional')
plt.xticks(np.arange(df.year.nunique())+bar_width/2,df.year.unique())
plt.legend()
plt.show()


# In[ ]:


dataAvgPrice = []
dataDate =[]
dataTotalVolume=[]
for i in df.Date.dt.year.unique():
    for j in reversed(df.Date.dt.month.unique()):
        dataAvgPrice.append(df[(df.Date.dt.year == i) & (df.Date.dt.month == j)].AveragePrice.mean())
        dataDate.append(i.astype('str')+' '+j.astype('str'))
        dataTotalVolume.append(df[(df.Date.dt.year == i) & (df.Date.dt.month == j)].Total_Volume.mean()/(10**6))
plt.subplots(figsize = (15,15))        
plt.plot(dataDate,dataAvgPrice,label = 'price')
plt.plot(dataDate,dataTotalVolume, label = 'volume/1e6')
plt.xticks(rotation = 'vertical')
plt.legend(loc = 'best')
plt.show()


# Both prices and volume increases in the long run. We can say that demand also increases.

# In[ ]:




