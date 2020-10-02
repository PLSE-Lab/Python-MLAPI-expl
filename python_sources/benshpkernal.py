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


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


test.info()


# In[ ]:


train.drop('MiscFeature', axis = 1, inplace = True)
test.drop('MiscFeature', axis = 1, inplace = True)


# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


sns.boxplot(x = "LotConfig", y= 'LotFrontage', data = train)


# In[ ]:


sns.boxplot(y="Fireplaces", x="FireplaceQu", data = test)


# In[ ]:


train["FireplaceQu"].count()


# In[ ]:


train["Fence"].value_counts()


# In[ ]:


train["Alley"].value_counts()


# In[ ]:


sns.jointplot(x = "LotFrontage", y = "SalePrice", data = train)


# In[ ]:


#go through and clean all data, remove outliers and NaNs. Go through columns and see what correlates well with Sale Price

