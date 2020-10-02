#!/usr/bin/env python
# coding: utf-8

# **Hello World Kernel**
# 
# This my first kernel that I write in Istanbul.
# 
# In this Kernel I will try to explain how you can write first Kernel in Kaggle while I am trying to do same for myself too.
# 
# **1. My Background**
# 
# I have already known about programming, algorithm structures and started learning Python two years ago. Then I could not achieve being sustainable to study. I failed in Python. Later I uploaded Anaconda.  And here we are again. This is the milestone for my Data Science career.
# 
# *Thanks to Datai Team on Udemy!*
# 
# I learned some basics until now like:
# 
# * Python Syntax (data types, functions, loops)
# * Skills of Numpy, Pandas, Matplotlib and Seaborn Libraries 
# * Basics of Data Science (Data Implementing & Filtering)
# * Introduction to Kaggle 
# 
# 
# **2. Dataset**
# 
# I will use [Heart Disease UCI](https://www.kaggle.com/ronitf/heart-disease-uci) in this Kernel.
# 
# 
# 

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


data = pd.read_csv('../input/heart-disease-uci/heart.csv')


# First look in data. What we have? Column names & Data Types

# In[ ]:


data.info()


# Data Correlation. Gives you correlation of data between discrete values. 
# 
# * Approaching 1 means datas are related each other positively. (Positive Correlation)
# * Approaching 0 means datas are not related each other.
# * Approaching 1 means datas are related each other negatively. (Negative Correlation)

# In[ ]:


data.corr()


# In[ ]:


#correlation map

f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# In[ ]:


data.head(7)


# In[ ]:


data.columns


# In[ ]:


# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.restecg.plot(kind = 'line', color = 'g',label = 'restecg',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data.oldpeak.plot(color = 'r',label = 'oldpeak',linewidth=1, alpha = 0.2,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()


# In[ ]:


# Scatter Plot 
# x = attack, y = defense
data.plot(kind='scatter', x='trestbps', y='chol',alpha = 0.5,color = 'red')
plt.xlabel('trestbps')              # label = name of label
plt.ylabel('chol')
plt.title('trestbps chol Scatter Plot')            # title = title of plot


# In[ ]:


# clf() = cleans it up again you can start a fresh
data.trestbps.plot(kind = 'hist',bins = 50)
plt.clf()
# We cannot see plot due to clf()


# In[ ]:


#create dictionary and look its keys and values
dictionary = {'turkey' : 'trabzon','georgia' : 'batumi'}
print(dictionary.keys())
print(dictionary.values())


# In[ ]:


# In order to run all code you need to take comment this line
# del dictionary         # delete entire dictionary     
print(dictionary)       # it gives error because dictionary is deleted


# In[ ]:


data = pd.read_csv('../input/heart-disease-uci/heart.csv')


# In[ ]:


series = data['exang']        # data['Defense'] = series
print(type(series))
data_frame = data[['exang']]  # data[['Defense']] = data frame
print(type(data_frame))


# In[ ]:


# Comparison operator
print(3 > 2)
print(3!=2)
# Boolean operators
print(True and False)
print(True or False)


# In[ ]:


# 1 - Filtering Pandas data frame
x = data['trestbps']>180     # There are only 2 patients who have higher defense value than 180
data[x]


# In[ ]:


# This is also same with previous code line. Therefore we can also use '&' for filtering.
data[(data['trestbps']>165) & (data['chol']>150)]

#filtering with pandas
#data[np.logical_and(data['Defense']>200, data['Attack']>100 )]   


# In[ ]:


# Stay in loop if condition( i is not equal 5) is true
i = 0
while i != 5 :
    print('i is: ',i)
    i +=1 
print(i,' is equal to 5')

