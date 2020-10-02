#!/usr/bin/env python
# coding: utf-8

# In[57]:


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


# In[58]:


df = pd.read_csv('../input/wc2018-players.csv')
df.head()


# In[59]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[60]:


df.Team.unique()


# In[61]:


temp = df[['Team','FIFA Popular Name','Height']].sort_values('Height',ascending=False)
print('tallest players and their team',temp.head())
print('\n')
print('shortest players and their team',temp.tail())


# In[62]:


temp = df[['Team','FIFA Popular Name','Weight']].sort_values('Weight',ascending=False)
print('high weight players and their team',temp.head())
print('\n')
print('low weight players and their team',temp.tail())


# In[63]:


print('total num of teams',len(df['Team'].unique()))


# In[64]:


print(df['Team'].value_counts())


# In[ ]:




