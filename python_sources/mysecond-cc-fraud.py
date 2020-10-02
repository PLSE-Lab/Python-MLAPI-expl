#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data=pd.read_csv('../input/creditcard.csv')


# In[ ]:


data.info()


# In[ ]:


data.corr()


# In[ ]:


data.describe()


# In[ ]:


data.head()


# In[ ]:



_,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# In[ ]:


#line plot
data.V1.plot(kind = 'line', color = 'g',label = 'V1',linewidth=1,alpha = 1,grid = False,linestyle = '--')
data.V2.plot(color = 'r',label = 'V2',linewidth=1, alpha = 0.5,grid = False,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()


# In[ ]:


# Scatter Plot 
data.plot(kind='scatter', x='V7', y='Amount',alpha = 0.5,color = 'blue')
plt.xlabel('V7')              # label = name of label
plt.ylabel('Amount')
plt.title('Scatter Plot')            # title = title of plot
plt.show()


# In[ ]:


# Histogram
# bins = number of bar in figure
data.Amount.plot(kind = 'hist',bins = 50,figsize = (5,5)) 
plt.show()


# In[ ]:


#since histogram shows a great deal of condensation, i wanted to descirbe the data so that i could see the std, mean and median
data.Amount.describe()


# In[ ]:


print(data.Amount.nlargest(5).tail(1))
print(data.Amount.nlargest(5).iloc[-1])


# In[ ]:


#filtering
myfilter=(data['Amount']>10) & (data['Amount']<1000)
data2=data[myfilter]
data2.Amount.describe()


# In[ ]:


#now it looks better after elimanting some values
data2.Amount.plot(kind = 'hist',bins = 50,figsize = (5,5)) 
plt.show()


# In[ ]:




