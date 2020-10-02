#!/usr/bin/env python
# coding: utf-8

# # My First Data Science Experience

# *  **In this tutorial, I'm doing my first data science experiments.**
# *  ***You can follow step by step notebook.***
# *  That's all for now. It will develop further in the future.
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns   # visualization tool
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data1 = pd.read_csv('../input/2015.csv')
data2 = pd.read_csv('../input/2016.csv')
data3 = pd.read_csv('../input/2017.csv')
# receive data / obtain data


# In[ ]:


print("Year 2015:")
data1.info()
print("\n Year 2016:")
data2.info()
print("\n Year 2017: ")
data3.info()
# we learn about the contents of the data


# In[ ]:


# Let's start processing 2017 data
data3.corr()


# In[ ]:


# correlation map
f,ax = plt.subplots(figsize = (21,21))
sns.heatmap(data3.corr(), annot= True, linewidths= .5, fmt= '.1f', ax= ax)
plt.show()


# In[ ]:


data3.head(24)
# Show 15 in our data from the beginning.


# In[ ]:


data3.columns
# What are the features of 2017 data?


# # Let's process this data graphically.

# In[ ]:


# Line plot
data3.Freedom.plot(kind= 'line', color= 'red', label= 'Freedom', linewidth= 1, alpha = 0.8, grid= True, linestyle= ':')
data3.Generosity.plot(kind= 'line', color= 'blue', label= 'Generosity', linewidth= 1, alpha = 0.8, grid= True, linestyle= '-.')
plt.legend(loc='upper right')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Line Plot')
plt.show()


# In[ ]:


# Scatter Plot
# correlation between two variables
data3.plot(kind='scatter', x= 'Freedom', y= 'Generosity', alpha = 0.6, color= 'green')
plt.xlabel('Freedom')
plt.xlabel('Generosity')
plt.title('Freedom-Generosity Scatter Plot')


# In[ ]:


# Histogram
# Let's learn about the frequency of feature
data3.Freedom.plot(kind= 'hist', bins= 65, figsize= (21,21))
plt.show()
# bins = number of bar in figure


# In[ ]:


# clf() = cleans it up again you can start a fresh
data3.Generosity.plot(kind= 'hist', bins =65)
plt.clf()
# we can't see plot due to clf()


# # ***Leave a small note here:***
# ***I go, You go, We go, xdxd, DATAI Team Kaan Can xd***

# In[ ]:


data3 = pd.read_csv('../input/2017.csv')
series = data3['Freedom']
print(type(series))
data_frame = data3[['Freedom']]
print(type(data_frame))


# In[ ]:


# With Pandas, let's make a dataframe filtering example
filter1 = data3['Economy..GDP.per.Capita.'] > 1.5
data3[filter1]


# In[ ]:


# we search for something more specific
data3[(data3['Happiness.Rank'] < 36) & (data3['Economy..GDP.per.Capita.'] > 1.6)]
# Are these the most livable countries?


# # Are these the most livable countries?
# ***Hmm... I don't know...***

# # We will continue here.
