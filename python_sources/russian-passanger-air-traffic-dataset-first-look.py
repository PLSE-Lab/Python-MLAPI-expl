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


df = pd.read_csv('/kaggle/input/russian-passenger-air-service-20072020/russian_passenger_air_service.csv')


# In[ ]:


df.head()


# **Lets take a look at summary air traffic by year**

# In[ ]:


pivot_df = pd.pivot_table(df, index = "Year", values = "Whole year", aggfunc = sum)
pivot_df


# In[ ]:


pivot_df.plot()


# Lets check in what month usualy have largest air traffic.

# In[ ]:


pivot_df_month = pd.pivot_table(df, index = "Year", values = ['January','February','March','April','May','June','July','August','September','October','November','December'], aggfunc = sum)
pivot_df_month = pivot_df_month.drop(2020, axis = 0)
pivot_df_month.loc['sum'] = pivot_df_month.sum(axis = 0)
pivot_df_month


# In[ ]:


pivot_df_month[pivot_df_month.index == 'sum'].plot.bar()
# it's december


# As you can see this data set may be usefull. Could you predict upcoming traffic or for example corona spread with it?
# 
# 
# If you liked data set pls upvote it.
