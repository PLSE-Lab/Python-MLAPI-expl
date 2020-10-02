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

from tqdm import tqdm_notebook
import matplotlib.pyplot as plt


# In[ ]:


sf_permits = pd.read_csv("../input/Building_Permits.csv")


# In[ ]:


sf_permits.sample(5)
print(sf_permits.shape)


# Checking # of NaNs across a) records ( x axis ) and b) across certain features ( y axis)

# In[ ]:


""" a) 
sf_permits.isnull().sum(axis = 1) """


# In[ ]:


# b) 
missing_values = sf_permits.isnull().sum(axis = 0)
total_missing_values = missing_values.sum()


# TOTAL %age of missing values

# In[ ]:


print((total_missing_values/np.product(sf_permits.shape)) * 100)


# If we remove all the columns with NaN,  then 31 columns information will be lost

# In[ ]:


col = sf_permits.dropna(axis = 1).columns
len([c for c in sf_permits.columns if c not in col])


# Lets try imputation
# 1. We fill all NaN with bfill method (use NEXT valid observation to fill gap) and any remaining NaN with 0

# In[ ]:


filled_na = sf_permits.fillna(method = 'bfill', axis = 0).fillna(0)


# Checking constant features

# In[ ]:


nunique = sf_permits.nunique(dropna = False)
nunique.sort_values()[:10]


# Checking duplicated features

# In[ ]:


sf_permits.fillna('NaN', inplace=True)
permit_enc = pd.DataFrame(index = sf_permits.index)


# In[ ]:


for col in tqdm_notebook(sf_permits.columns):
    permit_enc[col] = sf_permits[col].factorize()[0]


# In[ ]:


dup_cols = {}

for i, c1 in enumerate(tqdm_notebook(permit_enc.columns)):
    for c2 in permit_enc.columns[i+1:]:
        if c2 not in dup_cols and np.all(permit_enc[c1] == permit_enc[c2]):
            dup_cols[c2] = c1


# In[ ]:


dup_cols


# Determine types

# In[ ]:


plt.figure(figsize=(14,6))
_ = plt.hist(nunique.astype(float)/sf_permits.shape[0], bins = 100)

