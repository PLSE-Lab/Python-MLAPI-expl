#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns #visualization took

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/pokemon-challenge/pokemon.csv')


# In[ ]:


data.info()


# In[ ]:


data.corr()


# In[ ]:


#correlation map
f,ax = plt.subplots(figsize=(18,18))
sns.heatmap(data.corr(), annot =True, linewidths = .5, fmt = '.1f', ax=ax)
plt.show()


# In[ ]:


data.head(10)


# In[ ]:


data.columns


# **1.Introduction To Python**
# 

# In[ ]:


# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.Speed.plot(kind = 'line',color='g',label='Speed',linewidth=1, alpha = 0.5, grid=True, linestyle=':')
data.Defense.plot(color='r',label='Defense',linewidth='1',alpha=0.5,grid=True,linestyle='-')
plt.legend(loc='upper right')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Line Plot')
plt.show()


# In[ ]:


#Scatter Plot
#x = attac, y = defense
data.plot(kind='scatter',x='Attack', y='Defense',alpha=0.5,color='red')
plt.xlabel('Attack')
plt.ylabel("Defence")
plt.title('Attack - Defence Scatter Plot')


# In[ ]:


# Histogram
# bins = number of bar in figure
data.Speed.plot(kind='hist', bins = 50, figsize=(15,15))
plt.show()


# In[ ]:


#clf() = cleans it up again you can start a fresh
data.Speed.plot(kind = 'hist', bins = 50)
plt.clf()
#We cannot see plot due to clf()


# **2. Dictionary**

# In[ ]:


#create dictionary and look its keys and values
dictionary = {'spain':'madrid','usa':'las vegas'}
print(dictionary.keys())
print(dictionary.values())


# In[ ]:


dictionary['spain'] = "barcelona"
print(dictionary)
dictionary['france']="paris"
print(dictionary)
del dictionary['spain']
print(dictionary)
print('france' in dictionary)
dictionary.clear()
print(dictionary)


# **3- Pandas**

# In[ ]:


data = pd.read_csv('../input/pokemon-challenge/pokemon.csv')


# In[ ]:


series = data['Defense']
print(type(series))
data_frame = data[['Defense']]
print(type(data_frame))


# **4-Filtering**
# 

# In[ ]:


x = data['Defense'] > 200
data[x]


# In[ ]:


data[np.logical_and(data['Defense'] > 200, data['Attack'] > 100)]


# In[ ]:


data[(data['Defense'] > 200) & (data['Attack'] > 100)]


# ***2-Python Data Science Toolbox***
# 

# **a-User Defined Functon**
# 

# In[ ]:


def tuble_ex():
    """return defined t tuble"""
    t = (1,2,3)
    return t
a,b,c= tuble_ex()
print(a,b,c)


# **Scope**

# In[ ]:


x = 2
def f():
    x = 3
    return x
print(x)
print(f())


# In[ ]:


x = 5
def f():
    y = 2*x
    return y
print(f())


# **Nested Function**

# In[ ]:


def square():
    def add():
        x=2
        y=3
        z=x+y
        return z
    return add()**2
print(square())


# **Default and Flexible Arguments**

# In[ ]:


def f(a,b=1,c=2):
    y = a +b +c
    return y
print(f(5))
print(f(5,4,3))

