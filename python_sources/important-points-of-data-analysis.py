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


# **Difference Between iloc and loc(Use in python)**

# If labels are given then use loc to extract rows or columns from a dataframe.
# 
# Syntex: data.loc[row selection, column selection]

# Here labels/index names are given for each column. So we can extract rows/columns using that

# In[2]:


df = pd.DataFrame([[1, 2], [4, 5], [7, 8]],
                  index=['cobra', 'viper', 'sidewinder'], 
                  columns=['max_speed', 'shield'])


# In[3]:


df


# In above dataframe cobra,viper,sidewinder are indexes and max_speed and shield are columns

# In[4]:


df.loc['viper']          #using index name columns are extracted


# In[5]:


df.loc[:,'max_speed']     # all rows are extracted of column 'max_speed'


# In[6]:


df


# In[7]:


df.loc['cobra':'viper', 'max_speed']


# In[8]:


df.loc['cobra', 'max_speed':'shield']


# #### rows-rows or column-column are seperated with each other by using :(colon)
# #### column-rows are seperated using , (comma)
# 

# use of iloc : iloc is used to extract column or rows by the position of row or column

# In[9]:


df.iloc[1:]


# In[10]:


df.iloc[:,1]


# In[ ]:




