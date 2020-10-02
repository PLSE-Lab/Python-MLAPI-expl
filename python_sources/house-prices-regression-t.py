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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# To print all rows n colums
pd.options.display.max_rows = 100
pd.options.display.max_columns = 100


# In[ ]:


df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
df.head()


# In[ ]:


# To print col names
df.columns


# In[ ]:


df.isna().sum()


# In[ ]:


df.count()


# In[ ]:


df.drop(["Alley","PoolQC","MiscFeature"],axis=1,inplace = True)
df.isna().sum()


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


#describe all about saleprice
df["SalePrice"].describe()


# In[ ]:


#to describe all numeric values
df.describe()


# In[ ]:


#Plotting graph
sns.distplot(df["SalePrice"])


# In[ ]:


#Find outlier
df[df.SalePrice >=450000]


# In[ ]:


df.LotFrontage.unique()

