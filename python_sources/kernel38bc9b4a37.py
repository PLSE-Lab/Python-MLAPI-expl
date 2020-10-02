#!/usr/bin/env python
# coding: utf-8

# In[32]:


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


# In[33]:


data=pd.read_csv('../input/employee_reviews.csv')


# In[34]:


data.info()


# In[35]:


data.dtypes


# In[36]:


data.head(15)


# In[37]:


print(data.company.value_counts(dropna=False))


# In[38]:


data.describe()


# In[39]:


data.boxplot(column='helpful-count')


# In[40]:


data_new=data.head(10)
melted=pd.melt(frame=data_new,id_vars='Unnamed: 0',value_vars=['overall-ratings','helpful-count'])
melted


# In[41]:


melted.pivot(index='Unnamed: 0',columns='variable',values='value')


# In[42]:


data1=data.head()
data2=data.tail()
conc_data_row=pd.concat([data1,data2],axis=0,ignore_index=True)
conc_data_row


# In[43]:


data3=data['work-balance-stars'].head(10)
data4=data['carrer-opportunities-stars'].head(10)
conc_data_col=pd.concat([data3,data4],axis=1)
conc_data_col


# In[44]:


data.dtypes


# In[45]:


data['work-balance-stars']=pd.to_numeric(data['work-balance-stars'],errors='coerce')  # I couldnt convert the object as you show(with astype). I got an error.                                                                         
data.dtypes


# In[49]:


data['work-balance-stars'].value_counts(dropna=False)
data['work-balance-stars'].dropna(inplace=True)
assert data['work-balance-stars'].notnull().all()

