#!/usr/bin/env python
# coding: utf-8

# # DATA SCIENTIST
# **In this tutorial, I only explain you what you need to be a data scientist neither more nor less.**
# 
# Data scientist need to have these skills:
# 
# 1. Basic Tools: Like python, R or SQL. You do not need to know everything. What you only need is to learn how to use **python**
# 1. Basic Statistics: Like mean, median or standart deviation. If you know basic statistics, you can use **python** easily. 
# 1. Data Munging: Working with messy and difficult data. Like a inconsistent date and string formatting. As you guess, **python** helps us.
# 1. Data Visualization: Title is actually explanatory. We will visualize the data with **python** like matplot and seaborn libraries.
# 1. Machine Learning: You do not need to understand math behind the machine learning technique. You only need is understanding basics of machine learning and learning how to implement it while using **python**.
# 
# ### As a summary we will learn python to be data scientist !!!

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')


# In[ ]:


data.info()


# In[ ]:


data.corr()


# In[ ]:


#correlation map
f,ax = plt.subplots(figsize=(16,16))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f', ax=ax)
plt.show()


# In[ ]:


data.head(10)


# In[ ]:


data.columns


# ### MATPLOTLIB
# 
# Matplot is a python library that help us to plot data. The easiest and basic plots are line, scatter and histogram plots.
# 
# * Line plot is better when x axis is time.
# 
# * Scatter is better when there is correlation between two variables
# 
# * Histogram is better when we need to see distribution of numerical data.
# 
# * Customization: Colors,labels,thickness of line, title, opacity, grid, figsize, ticks of axis and linestyle

# In[ ]:


# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.Deaths.plot(kind = 'line', color = 'b',label = 'Deaths',linewidth=3,alpha = 0.9,grid = True,linestyle = ':')
data.Confirmed.plot(color = 'r',label = 'Confirmed',linewidth=2, alpha = 0.6,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()


# In[ ]:


# Scatter Plot 
# x = Confirmed, y = Deaths
data.plot(kind='scatter', x='Confirmed', y='Deaths',alpha = 0.5,color = 'red')
plt.xlabel('Confirmed')              # label = name of label
plt.ylabel('Deaths')
plt.title('Confirmed Deaths Scatter Plot')            # title = title of plot
plt.show()


# In[ ]:


# Histogram
# bins = number of bar in figure
data.Confirmed.plot(kind = 'hist',bins =10,figsize = (8,8))
plt.show()


# In[ ]:


# clf() = cleans it up again you can start a fresh
data.Confirmed.plot(kind = 'hist',bins = 10)
plt.clf()
# We cannot see plot due to clf()


# In[ ]:


# 1 - Filtering Pandas data frame
x = data['Deaths']>10000     
data[x]


# In[ ]:


# For pandas we can achieve index and value
for index,value in data[['Country/Region']][0:10].iterrows():
    print(index," : ",value)


# In[ ]:


# For example lets look frequency of Covid-19 types
print(data['Country/Region'].value_counts(dropna =False)) 


# In[ ]:


# For example max HP is 255 or min defense is 5
data.describe() #ignore null entries


# In[ ]:




