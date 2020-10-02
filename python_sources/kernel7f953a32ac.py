#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


d=pd.read_csv('/home/mahima/Desktop/bitcoinData.csv')


# In[ ]:


d.head()


# 

# In[ ]:


d.tail()


# 

# In[ ]:


d.shape


# In[ ]:


d.shape


# In[ ]:


d.count()


# In[ ]:


d.describe()


# In[ ]:


d['Date']=pd.to_datetime(d['Date'])


# In[ ]:


d.dtypes


# In[ ]:


data=d.set_index("Date")
data.head()


# In[ ]:


d.corr()


# In[ ]:


data['avg']=(data['Open*']+data['Close**'])/2
data.head()


# In[ ]:


x=data['Open*']
y=data['Close**']
plt.figure(figsize=(15,8))
plt.plot(x,color='r')
plt.plot(y,color='b')
plt.show()


# In[ ]:


plt.figure(figsize=(15,8))
plt.plot(data.index,data.avg)
plt.title('data')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




