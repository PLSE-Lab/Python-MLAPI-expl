#!/usr/bin/env python
# coding: utf-8

# In[6]:


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


# Load data from each csv into its own dataframe:

# In[8]:


zdahi1 = '../input/zdahi1.csv'
zdahi2 = '../input/zdahi2.csv'
df1 = pd.read_csv(zdahi1)
df2 = pd.read_csv(zdahi2)


# Print the head of each dataframe:

# In[9]:


print (df1.head())
print (df2.head())


# In[15]:


import seaborn as sns
g = sns.pairplot(df2, hue="Chr")

