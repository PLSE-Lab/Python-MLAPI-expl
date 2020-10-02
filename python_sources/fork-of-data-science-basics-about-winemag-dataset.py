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
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data_wine = pd.read_csv('../input/winemag-data_first150k.csv')


# Basic informations about this dataset.
# 

# In[ ]:


data_wine.info()


# In[ ]:


data_wine.corr()


# In[ ]:


f , ax = plt.subplots(figsize=(14,14))
sns.heatmap(data_wine.corr() , annot=True , linewidths=0.6 , fmt='.1f' , ax=ax)
plt.show() 


# In[ ]:


data_wine.head()


# scatter

# In[ ]:


#data_wine.columns
plt.scatter(data_wine.index , data_wine.price , color='red' , alpha=0.5)
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Price')


# histogram
# 

# In[ ]:


data_wine.price.plot(kind='hist',bins=50 , figsize=(5,5))
plt.show()


# pandas

# In[ ]:


x=data_wine['price']>1500

data_wine[x].head()


# filtering

# In[ ]:


data_wine[np.logical_and(data_wine['country']=='US' , data_wine['price']>500)]


# filtering with & operator

# In[ ]:


c=data_wine[(data_wine['country']=='France') & (data_wine['price']>500)]
c


# scope

# In[ ]:


x=5
def g():
    y=2**x
    return y
print(g())


# nested function
# 

# In[ ]:


def volume():
    def add():
        x=3
        y=4
        z=x+y
        return z
    return add()**3
print(volume())
    


# In[ ]:


def f(a,b=5,c=7):
    h=a+b+c
    return h
print(f(3))
print(f(4,6,8))


# In[ ]:


def f(*args):
    for i in args:
        print(i)  
f(1,2,3,4)


# In[ ]:


z= lambda x,y : y+x
print(z(4,5))


# In[ ]:


number_list=[2,3,5]
z = map(lambda x:x**2,number_list)
print(list(z))

