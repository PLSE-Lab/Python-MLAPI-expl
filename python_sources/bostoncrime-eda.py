#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas import Series,DataFrame
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


df = pd.read_csv("../input/crime.csv",encoding='ISO-8859-1')
df.head()


# In[ ]:


df.info()


# In[ ]:


df['YEAR'].value_counts()


# In[ ]:


df['DISTRICT'].value_counts()


# In[ ]:


df['MONTH'].value_counts()


# In[ ]:


df['SHOOTING'].value_counts()


# Wherever shooting is not given as 'Y' , I am  assuming there was no shooting involved.  
# Hence, I am filling them with 'N'.

# In[ ]:


df['SHOOTING'].fillna('N',inplace=True)
df['SHOOTING'].value_counts()


# The number of crimes involving shooting is much less than that not involving.  
# I am dividing up the dataset on this basis.

# In[ ]:


shoot_grp = df.groupby('SHOOTING')
df_y = shoot_grp.get_group('Y')
df_n = shoot_grp.get_group('N')


# In[ ]:


fig,ax = plt.subplots(1,2,figsize=(16,6))
sns.countplot(x='YEAR',data=df_n,ax=ax[0])
sns.countplot(x='YEAR',data=df_y,ax=ax[1])
plt.show()


# In[ ]:


fig,ax = plt.subplots(1,2,figsize=(16,6))
sns.countplot(x='MONTH',data=df_n,ax=ax[0])
sns.countplot(x='MONTH',data=df_y,ax=ax[1])
plt.show()


# In[ ]:


fig,ax = plt.subplots(1,2,figsize=(16,6))
sns.countplot(x='DAY_OF_WEEK',data=df_n,ax=ax[0])
sns.countplot(x='DAY_OF_WEEK',data=df_y,ax=ax[1])
plt.show()


# In[ ]:


fig,ax = plt.subplots(1,2,figsize=(16,6))
sns.countplot(x='HOUR',data=df_n,ax=ax[0])
sns.countplot(x='HOUR',data=df_y,ax=ax[1])
plt.show()


# In[ ]:


df['OFFENSE_CODE_GROUP'].value_counts()


# **Top 20 most common types of crimes in Boston**

# In[ ]:


oc_count = df['OFFENSE_CODE_GROUP'].value_counts()
oc_top = oc_count[:25]
plt.figure(figsize=(10,5))
plt.bar(oc_top.index,oc_top)
plt.xticks(rotation=90)
plt.show()


# **Number of crimes by district**

# In[ ]:


fig,ax = plt.subplots(1,2,figsize=(16,6))
sns.countplot(x='DISTRICT',data=df_n,ax=ax[0])
sns.countplot(x='DISTRICT',data=df_y,ax=ax[1])
plt.show()


# In[ ]:


fig,ax = plt.subplots(1,2,figsize=(16,6))
sns.countplot(x='UCR_PART',data=df_n,ax=ax[0])
sns.countplot(x='UCR_PART',data=df_y,ax=ax[1])
plt.show()


# In[ ]:




