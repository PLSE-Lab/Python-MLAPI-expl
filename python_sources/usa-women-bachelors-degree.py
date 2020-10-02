#!/usr/bin/env python
# coding: utf-8

# I would just like to analyze the data and find some insights.

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


df = pd.read_csv("../input/percent-bachelors-degrees-women-usa.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# I would like to compare the growth for every 5 years

# In[31]:


def group_by_5_years(value):
    if value < 1976 :
        return "71-76"
    elif value < 1981:
        return "76-81"
    elif value < 1986:
        return "81-86"
    elif value < 1991:
        return "86-91"
    elif value < 1996:
        return "91-96"
    elif value < 2001:
        return "96-01"
    elif value < 2006:
        return "01-06"
    else:
        return "06-11"


# In[7]:


df.Year = df.Year.apply(group_by_5_years)


# In[8]:


sns.barplot(x="Year",y="Agriculture",data = df)


# In[10]:


group = df.groupby("Year").mean().reset_index()


# In[24]:


def plotmatrix(start,end):
    fig, axs = plt.subplots(nrows = 2, ncols=2)
    i = 0
    cols = df.columns[start:end]
    fig.set_size_inches(14, 10)
    for indi in range(2):
        for indj in range(2):
                sns.barplot(x="Year",y=str(cols[i]),data = group,ax = axs[indi][indj],                            order = ['71-76', '76-81', '81-86', '86-91', '91-96','96-01','01-06','06-11'])
                i+=1
                #plt.xticks(rotation = 90)


# In[25]:


plotmatrix(1,5)


# So from the above fig, it's clear that
# 1. Growth is in increasing order in Agriculture Field and even in Architecture Field
# 2. In Art and Performance, the growth is almost stagnant.
# 3. It's interesting to see that growth for Biology is declining from 2001-06 to 2006-11. 

# In[32]:


plotmatrix(6,10)


# It's clearly understood that, women growth in Compute Science is declining

# In[33]:


plotmatrix(11,15)


# (under editing)
