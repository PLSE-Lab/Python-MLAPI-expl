#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
#This is very simple project since I am new at this place ,I wanted to show the distribution of wins,loses and draws of hometowns
# in international football from 1872 to 2019
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


data=pd.read_csv('../input/international-football-results-from-1872-to-2017/results.csv')
#reading data


# In[ ]:


data.info()
#checking...


# In[ ]:


data.drop(['home_team','away_team','tournament','city','country','neutral'], axis = 1, inplace = True)
#get rid of unnecessary titles


# In[ ]:


data.corr()
#checking again


# In[ ]:


w=data.home_score>data.away_score
#determine wins


# In[ ]:


data[w].head()


# In[ ]:


data[w].tail()


# In[ ]:


l=data.home_score<data.away_score
#determine loses


# In[ ]:


data[l].head()


# In[ ]:


data[l].tail()


# In[ ]:


d=data.home_score==data.away_score
#determine draws


# In[ ]:


data[d].head()


# In[ ]:


data.tail()


# In[ ]:


data[w].plot(kind ="line",color='green',label='Attack',linewidth=5,alpha = 0.5,grid = True,linestyle = ':')
plt.xlabel('Years')
plt.ylabel('Counted Hometown Wins')
data[l].plot(kind ="line",color='red',label='Attack',linewidth=5,alpha = 0.5,grid = True,linestyle = ':')
plt.xlabel('Years')
plt.ylabel('Counted Hometown Loses')
data[d].plot(kind ="line",color='yellow',label='Attack',linewidth=5,alpha = 0.5,grid = True,linestyle = ':')
plt.xlabel('Years')
plt.ylabel('Counted Hometown Draws')

#Looking distribution of wins,loses and draws of hometown by years
#Since this is my first project and I am new at Kaggle,I know it is very easy and simple project :)

