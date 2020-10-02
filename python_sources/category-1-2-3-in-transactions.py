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


# Categories in transactions files are engineered features from geographic information or how payments are made.

# In[ ]:


df = pd.read_csv('../input/historical_transactions.csv')
dnew = pd.read_csv('../input/new_merchant_transactions.csv')
df = pd.concat([df,dnew])


# In[ ]:


print(set(df.loc[df['category_3']=='A','installments']))


# In[ ]:


print(set(df.loc[df['category_3']=='B','installments']))


# In[ ]:


print(set(df.loc[df['category_3']=='C','installments']))


# Category 3 is based on how the payment is made.

# In[ ]:


print(set(df.loc[df['category_2']==1,'state_id']))


# In[ ]:


print(set(df.loc[df['category_2']==2,'state_id']))


# In[ ]:


print(set(df.loc[df['category_2']==3,'state_id']))


# In[ ]:


print(set(df.loc[df['category_2']==4,'state_id']))


# In[ ]:


print(set(df.loc[df['category_2']==5,'state_id']))


# Category 2 is based on the state_id. 

# In[ ]:


print(set(df.loc[df['category_1']=='Y','city_id']))


# In[ ]:


print(set(df.loc[df['category_1']=='N','city_id']))


# Category 1 is influenced by the city_id.
