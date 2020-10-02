#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/2016.csv')
data.info()


# In[ ]:


data.head(15)


# In[ ]:


data_cols = data.columns
data_cols = data_cols.str.replace(' ','_')
data_cols


# In[ ]:


data.columns = data_cols
data.columns


# **User defined function**

# In[ ]:


def calcCorr(a):
    return a.corr()


# **An example of Nested Function**

# In[ ]:


def plotHeatmap(a):
    f,ax = plt.subplots(figsize = (15,15))
    sns.heatmap(calcCorr(a), annot = True, linewidth=.6, fmt='.1f', ax=ax)
    plt.show()


# In[ ]:


calcCorr(data)
plotHeatmap(data)


# **Local and Global Scope**

# In[ ]:


x = 2
def f():
    x = 3
    return x
print(x)      # x = 2 global scope
print(f())    # x = 3 local scope


# **Built-in Scope**

# In[ ]:


# How can we learn what is built in scope
import builtins
dir(builtins)


# **Lambda Function**

# In[ ]:


pow5= lambda x: x**5     # where x is name of argument
print(pow5(4))
tot = lambda x,y,z: x+y+z   # where x,y,z are names of arguments
print(tot(12,54,66))


# **Iretation Process**

# In[ ]:


name = 'Atilla'
it = iter(name)
print(it)
print(next(it))
print(*it)


# ***Zip and Unzip***

# In[ ]:


zippedCols = zip(data.Family, data.Happiness_Score)
print(zippedCols)
zippedCols_list = list(zippedCols)
print(zippedCols_list)


# In[ ]:


un_zip = zip(*zippedCols_list)
un_list1,un_list2 = list(un_zip) # unzip returns tuple
print(un_list1)
print(un_list2)
print(type(un_list2))


# **Flexible arguments *args**

# In[ ]:


def f(*args):
    for i in args:
        print(i)
f(1)
print("")
f(1,2,3,4)


# **Flexible arguments with dictionary data type(how to use kwargs)**

# In[ ]:


def f(**kwargs):
    """ print key and value of dictionary"""
    for key, value in kwargs.items():               
        print(key, " ", value)
f(country = 'spain', capital = 'madrid', population = 1234560)
f(country = 'turkey', capital = 'ankara', population = 1231233, altitude = 1100)


# **Example of list comprehension**

# In[ ]:


num1 = [1,2,3]
num2 = [i + 1 for i in num1 ]
decision = [print("doing ok") if(len(num2)==3) else print("something is wrong!")]
print(num2)

