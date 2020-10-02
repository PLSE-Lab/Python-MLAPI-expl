#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[17]:


data = pd.read_csv('../input/Iris.csv')


# In[8]:


data.info()


# In[9]:


data.corr()


# In[10]:


f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.8, fmt= '.1f',ax=ax)
plt.show()


# In[11]:


data.head(10)


# In[25]:


data.columns


# MATPLOTLIB

# In[26]:


iris.PetalLengthCm.plot(kind = 'line', color = 'g',label = 'PetalLengthCm',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
iris.PetalWidthCm.plot(color = 'r',label = 'PetalWidthCm',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.xlabel('x axis')        
plt.ylabel('y axis')
plt.title('Line Plot')  
plt.show()


# In[24]:


iris.plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm',alpha = 0.5,color = 'green')
plt.xlabel('PetalLengthCm')              
plt.ylabel('PetalWidthCm')
plt.title('PetalLengthCm PetalWidthCm Scatter Plot')


# In[28]:


iris.PetalLengthCm.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()


# In[ ]:


iris.PetalWidthCm.plot(kind = 'hist',bins = 60,figsize = (12,12))
plt.show()


# In[ ]:


series = iris['PetalLengthCm'] 
print(type(series))
data_frame = data[['PetalWidthCm']] 
print(type(data_frame))


# In[ ]:


x = iris['PetalLengthCm']>1.4
iris[x]


# In[ ]:


iris[(iris['PetalLengthCm']>1.4) & (iris['PetalWidthCm']>0.2)]

