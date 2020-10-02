#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import seaborn as sns

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data1 = pd.read_csv("../input/chennai_reservoir_levels.csv")
data2 = pd.read_csv("../input/chennai_reservoir_rainfall.csv")


# In[ ]:


data1.info()


# In[ ]:


data2.info()


# In[ ]:


data2.describe()


# In[ ]:


data2.corr()


# In[ ]:


f,ax = plt.subplots(figsize=(8,8))
sns.heatmap(data1.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# In[ ]:


data1.head(n=5648)


# In[ ]:


data1.POONDI.plot(kind = 'line', color = 'g',label = 'POONDI',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data1.REDHILLS.plot(color = 'r',label = 'REDHILLS',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     
plt.xlabel('x axis')              
plt.ylabel('y axis')
plt.title('REDHILLS AND POONDI Plot')           
plt.show()


# In[ ]:


data1.plot(kind='scatter', x='REDHILLS', y='POONDI',alpha = 0.5,color = 'blue')
plt.xlabel('REDHILLS')              # label = name of label
plt.ylabel('POONDI')
plt.title('POONDI AND REDHILLS Scatter Plot')
plt.show()


# In[ ]:


data1.POONDI.plot(kind = 'hist',bins = 500,figsize = (18,18))
plt.show()


# In[ ]:


data1.REDHILLS.plot(kind = 'hist',bins = 500,figsize = (18,18))
plt.show()


# In[ ]:





# In[ ]:


data1.POONDI.plot(kind = 'line', color = 'g',label = 'POONDI',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data1.REDHILLS.plot(color = 'r',label = 'date',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     
plt.xlabel('x axis')              
plt.ylabel('y axis')
plt.title('DATE AND POONDI Plot')           
plt.show()

