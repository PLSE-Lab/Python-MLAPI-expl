#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for visualization
import seaborn as sns # for visualization

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/2017.csv') # read the csv file


# In[ ]:


data.info() # To understand what kind of datatypes this dataset has


# In[ ]:


data.columns # To see the titles of our dataset


# In[ ]:


data.head() # To be able to have a quick view about the dataset. It is 5 by default, you can change it. For instance, data.head(10).


# In[ ]:


data.describe() # Numerical values of the dataset


# In[ ]:


data.corr() # Corrolation values between our data inputs.


# With this code below, we can easily visualize our data. This is the one of the most useful visialization methods for data science.

# In[ ]:


# Correlation Map

f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# In[ ]:


# Line Plot
# kind: Type of your plot, color: Color of the values, alpha: Opacity of the values,
# grid: Background grid, linewidth: Width of line, linestyle: Linestyle,
# figsize: allows you to chage your plot's size

data.plot(kind = 'line', x='Happiness.Rank', y='Health..Life.Expectancy.', color = 'g', linewidth=1,alpha = 0.7, grid = True, linestyle = '-.', figsize=(20,5))
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('Happiness Rank')              # label = name of label
plt.ylabel('Health-Life Expectancy')
plt.title('Line Plot')            # title = title of plot
plt.show()


# In[ ]:


data1 = data['Happiness.Rank']
data2 = data['Happiness.Score']

plt.subplots(figsize=(15,10))
plt.plot(data1,data2,label = 'Happiness Rank-Score')
plt.legend()
plt.xlabel('Happiness Rank')
plt.ylabel('Happiness Score')
plt.title('Happiness Report of The World')
plt.show()


# In[ ]:


# Scatter Plot 
plt.subplots(figsize=(20,10))
plt.plot(data['Happiness.Rank'], data['Trust..Government.Corruption.'], color='r', label='Goverment Trust')
plt.plot(data['Happiness.Rank'], data['Health..Life.Expectancy.'], color='b', label='Health Life Expectancy')
plt.legend()
plt.title('Goverment Trust-Health Life Expectancy')
plt.xlabel('Happiness Rank')
plt.show()


# In[ ]:


data1 = data['Happiness.Rank']
data2 = data['Happiness.Score']
data3 = data['Trust..Government.Corruption.']

plt.subplots(figsize=(10,5))
plt.plot(data1,data2, label='Happiness score')
plt.plot(data1,data3, label='Goverment Trust')
plt.title('Goverment Trust and Happiness Score Connection')
plt.legend()
plt.xlabel('Happiness Rank')
plt.show()


# In[ ]:


# Histogram
# bins = number of bar in figure, rwidth = closeness of bars
data.Freedom.plot(kind = 'hist',bins = 25,figsize = (15,5), rwidth=0.9)
plt.title('Freedom')
plt.show()


# In[ ]:


# Bar Plot
plt.subplots(figsize=(20,10))
plt.bar(data['Country'][0:10], data['Economy..GDP.per.Capita.'][0:10],label='Economy Per Capita', color='r', width=0.5)
plt.legend()
plt.xlabel('Countries')
plt.title('Economy In Countries')
plt.show()


# In[ ]:


# Bar Plot
plt.subplots(figsize=(15,5))
plt.bar(data['Country'][0:10], data['Generosity'][0:10], label='Generosity')
plt.legend()
plt.xlabel('Countries')
plt.title('Generosity of The Countries')
plt.show()


# In[ ]:


plt.subplot(3,1,1)
plt.plot(data['Happiness.Rank'],data['Happiness.Score'],label = 'Happines Rank', color='orange')
plt.title('Rank-Score-Economy-Life Graph')
plt.grid()
plt.legend()
plt.subplot(3,1,2)
plt.plot(data['Happiness.Rank'],data['Economy..GDP.per.Capita.'],label = 'Economy..GDP.per.Capita.',color = 'b')
plt.grid()
plt.legend()
plt.subplot(3,1,3)
plt.plot(data['Happiness.Rank'],data['Health..Life.Expectancy.'],label = 'Health..Life.Expectancy.',color = 'r')
plt.grid()
plt.legend()
plt.show()

