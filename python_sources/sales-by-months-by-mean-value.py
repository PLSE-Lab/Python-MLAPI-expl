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
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data=pd.read_csv('../input/salesbymonths/satislar.csv')


# In[ ]:


data.shape


# In[ ]:


#Checking for missing data
data.columns
#"Aylar" means months and "Satislar" means sales


# In[ ]:


#Checking for missing data
data.info()


# In[ ]:


data.corr()


# In[ ]:


#Finding mean value for sales
Mean=data.Satislar.mean()


# In[ ]:


data.Satislar.plot(kind='line',color='Blue',Alpha=0.5,figsize=(9,7),label='Sales')
plt.plot([0,35],[Mean,Mean],label='Average',color='Red')
plt.legend()
plt.xlabel('Months')
plt.ylabel('Sales')
plt.grid(True)


# In[ ]:


#Looking for the first year
data12=data.head(12)


# In[ ]:


Mean12=data12.Satislar.mean()


# In[ ]:


data12.Satislar.plot(kind='line',color='Blue',Alpha=0.5,figsize=(9,7),label='Sales')
plt.plot([0,35],[Mean12,Mean12],label='Average',color='Red')
plt.legend()
plt.xlabel('Months')
plt.ylabel('Sales')
plt.grid(True)


# In[ ]:


fig = plt.figure(figsize = (18,6))
sns.barplot(x = 'Aylar', y = 'Satislar', data = data)
plt.xlabel('Months')
plt.ylabel('Sales')
plt.plot([0,24],[Mean,Mean],label='Average',color='Red')
plt.legend()
plt.grid(True)
plt.title("Sales by months")
plt.show()


# In[ ]:


data1=data[data.Satislar>data.Satislar.mean()]
data2=data[data.Satislar<data.Satislar.mean()]
data2.Satislar=(data2.Satislar-data.Satislar.mean())
data1.Satislar=(data1.Satislar-data.Satislar.mean())
#Making "zero point" as average value


# In[ ]:


data3=pd.concat([data1,data2])
#combinin datas


# In[ ]:


fig = plt.figure(figsize = (18,6))
sns.barplot(x = 'Aylar', y = 'Satislar', data = data3)
plt.xlabel('Months')
plt.ylabel('Sales')
plt.grid(True)
plt.title("Sales by months according to mean value")
plt.show()


# In[ ]:




