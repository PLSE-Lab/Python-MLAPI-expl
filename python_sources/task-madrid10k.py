#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('/kaggle/input/madrid-10k-new-years-eve-popular-race/madrid_10k_20191231.csv')


# In[ ]:


df.head()


# In[ ]:


df['2.5km_seconds'].isna().sum()


# In[ ]:


df['5km_seconds'].isna().sum()


# In[ ]:


df['7.5km_seconds'].isna().sum()


# In[ ]:


df['total_seconds'].isna().sum()


# In[ ]:


import seaborn as sns


# In[ ]:


df['age_category'].unique()


# * young: 16-34
# * old: 55+

# In[ ]:


age_compare_times = df[(df['age_category']=='16-19') | (df['age_category']=='20-22') | (df['age_category']=='55+')]


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


def age_color(ls, color):
    c =[]
    for i in ls:
        if(i=='16-19' or i=='20-22'):
            c.append(color[0])
        elif(i=='55+'):
            c.append(color[1])
        else:
            c.append(color[2])
    return c
        
            

plt.scatter(age_compare_times['id_number'],age_compare_times['total_seconds'], color = age_color(age_compare_times['age_category'],['g','r','w']))


# In[ ]:


total = len(age_compare_times)


# In[ ]:


old = len(age_compare_times[age_compare_times['age_category']=='55+'])


# In[ ]:


young = total - old
young


# young and old people are almost same in number

# In[ ]:


age_compare_times[age_compare_times['age_category']=='55+']['total_seconds'].mean()


# In[ ]:


age_compare_times[(age_compare_times['age_category']=='16-19') | (age_compare_times['age_category']=='20-22')]['total_seconds'].mean()


# * if fitness is defined as who completes race faster, **young** people are **fitter** than **older** people :D

# In[ ]:


age_compare_times.head()


# In[ ]:


df[df['2.5km_seconds'].isna()].head()


# In[ ]:


import numpy as np


# In[ ]:


df['2550_diff']=df['5km_seconds']-df['2.5km_seconds']
df['5075_diff']=df['7.5km_seconds']-df['5km_seconds']
df['7510_diff']=df['total_seconds']-df['7.5km_seconds']


# In[ ]:


df[df['7510_diff']<0]


# * HOW CAN TOTAL_SECONDS BE LESS THAN 7.5KM_SECONDS?
# * **POSSIBILITY OF CHEATING??**

# In[ ]:


df[(df['7510_diff']<0) & ((df['2.5km_seconds']).isna() | (df['5km_seconds']).isna() | (df['7.5km_seconds'].isna()))]


# * WILL DO FURTHER PART BASED ON CONFIRMATION OF THIS POSSIBLE ERROR.
