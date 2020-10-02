#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')
df


# In[ ]:


df.columns


# In[ ]:


df.isnull().any()


# In[ ]:


df.YearBuilt


# In[ ]:


df.YearBuilt.isnull()


# In[ ]:


any([True, True, True]), any([True, False, True]), any([False, False, False])


# In[ ]:


all([True, True, True]), all([True, False, True]), all([False, False, False])


# In[ ]:


df.isnull().sum()


# In[ ]:


df.shape


# In[ ]:


for col in df.columns:
    if df[col].isnull().any():
        print(col, df[col].isnull().sum())


# In[ ]:


for col in df.columns:
    if df[col].dtype == 'object':
        print(col)


# In[ ]:


for col in df.columns:
    if df[col].dtype == 'int64' or df[col].dtype == 'float64':
        print(col)


# In[ ]:


# Get list of categorical variables
s = (df.dtypes == 'object')
print(s)
object_cols = list(s[s].index)

print("Categorical variables:")
print(object_cols)


# In[ ]:


df_num = df.select_dtypes(exclude='object')
df_num


# In[ ]:


df_cat = df[object_cols]
df_cat


# In[ ]:




