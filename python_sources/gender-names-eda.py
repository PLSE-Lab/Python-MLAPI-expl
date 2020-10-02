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

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[3]:


male = (pd.read_csv("../input/Indian-Male-Names.csv"))
female = (pd.read_csv("../input/Indian-Female-Names.csv"))


# In[4]:


male.head()


# In[24]:


male = (male.assign( firstname = lambda x : x.name.str.split(' ').str[0],
    tri_last = lambda x : x.firstname.str[-3:],
    bi_last = lambda x : x.firstname.str[-2:]))
female = (female.assign( firstname = lambda x : x.name.str.split(' ').str[0],
    tri_last = lambda x : x.firstname.str[-3:],
    bi_last = lambda x : x.firstname.str[-2:]))


# In[25]:


female.head()


# ## Plotting top 10 bi_last and tri_last. covers a plurality of cases

# In[28]:


female['tri_last'].value_counts().head(10).plot('bar')


# In[29]:


female['bi_last'].value_counts().head(10).plot('bar')


# In[31]:


male['tri_last'].value_counts().head(10).plot('bar')


# In[32]:


male['bi_last'].value_counts().head(10).plot('bar')


# In[ ]:




