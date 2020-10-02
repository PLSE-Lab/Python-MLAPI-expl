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

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[3]:


from pandas.plotting import scatter_matrix
import matplotlib
import seaborn as sbn
import scipy as spy
from datetime import datetime


# In[4]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


equake_df = pd.read_csv('../input/eartquakes_Romania.csv')
print(equake_df.shape)
equake_df.head()


# In[6]:


cols = equake_df.columns
cols


# In[7]:


equake_df.describe()    


# In[8]:


print('Type:')
print(equake_df['type'].value_counts())
print('\n \n Status:')
print(equake_df['status'].value_counts())
print('\n \n Net:')
print(equake_df['net'].value_counts())


# The columns **type** , **status** and **net** have a single category value **earthquake** , **reviewed** and **us** respectively.
# Hence removing these 3 columns.

# In[8]:


equake_df.drop(['type','status','net'], axis=1, inplace=True)
equake_df.head()


# In[9]:


print(equake_df['locationSource'].value_counts())
print(equake_df['magSource'].value_counts())


# In[10]:


equake_df['loc_us'] = equake_df.locationSource.str.startswith('us').astype(int)
equake_df['loc_buc'] = (equake_df.locationSource=='buc').astype(int)
equake_df['loc_irsa'] = (equake_df.locationSource=='irsa').astype(int)
equake_df['loc_others'] = ((equake_df.locationSource!='irsa') & (equake_df.locationSource!='buc') & (~(equake_df.locationSource.str.startswith('us')))).astype(int)


# In[11]:


equake_df['magS_us'] = equake_df.magSource.str.startswith('us').astype(int)
equake_df['magS_buc'] = (equake_df.magSource=='buc').astype(int)
equake_df['magS_irsa'] = (equake_df.magSource=='irsa').astype(int)
equake_df['magS_others'] = ((equake_df.magSource!='irsa') & (equake_df.magSource!='buc') & (~(equake_df.magSource.str.startswith('us')))).astype(int)


# In[12]:


equake_df.drop(['locationSource', 'magSource'], axis=1, inplace=True)


# In[13]:


equake_df['updated'] = pd.to_datetime(equake_df['updated'], format="%Y-%m-%dT%H:%M:%S.%fZ")
equake_df['time'] = pd.to_datetime(equake_df['time'], format="%Y-%m-%dT%H:%M:%S.%fZ")
equake_df.dtypes


# In[ ]:


#fig = plt.figure(figsize=(12,8))
#plt.subplot(111)
#plt.bar(equake_df['mag'])
#plt.show()


# In[14]:


scatter_matrix(equake_df, alpha=0.4, figsize=(15,12), diagonal='kde');


# In[ ]:


for col in equake_df.columns:
    if len(equake_df[col].value_counts()) <= 20:
        print(col)


# In[10]:


print(equake_df['magType'].value_counts())


# In[ ]:


print(equake_df['place'].value_counts())


# In[ ]:


def location_parse(place):
    if 'border' not in place:
        place=place.split(',')[-1].strip()
    else:
        place = 'Romania border'
    return place        


# In[ ]:


def state_parse(place):
    if 'border' not in place:
        place=place.split('of')[-1].strip()
    else:
        place = 'Romania border'
    return place   


# In[ ]:


equake_df['parsed_state'] = equake_df['place'].apply(state_parse)
equake_df['parsed_state'].value_counts()


# In[ ]:


equake_df['parsed_country'] = equake_df['place'].apply(location_parse)
equake_df['parsed_country'].value_counts()


# In[ ]:


equake_df['magType'].unique()


# In[ ]:


equake_df.dtypes


# In[ ]:


equake_df['time'].head()


# In[ ]:


equake_df['ordinal_time'] = equake_df['time'].apply(lambda x: x.toordinal())
equake_df['time_year'] = equake_df['time'].apply(lambda x: x.year)
equake_df['time_month'] = equake_df['time'].apply(lambda x: x.month)
equake_df['time_day'] = equake_df['time'].apply(lambda x: x.day)


# In[ ]:


equake_df.ix[1,'time'].isoformat()


# In[ ]:


equake_df['time_year'].value_counts()


# In[ ]:


equake_df['time_month'].value_counts()


# In[ ]:


equake_df['time_day'].value_counts()


# In[ ]:


# code to get the date in year-month format and find the value_counts and start the visualization part of EDA

