#!/usr/bin/env python
# coding: utf-8

# # Just an attempt in answering an interesting question that was posted
# ## Please pardon if the heading and content is in accurate 
# ## All the credit goes to https://www.kaggle.com/general/65351

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

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


macfail = pd.read_csv('../input/machine-failure-data/machine_failure_data.csv')
macfail.head()


# In[ ]:


macfail.isna().sum()


# In[ ]:


# Extracting only the records based on NA threshold
macfail = macfail.drop(macfail.columns[macfail.apply(lambda col: col.isna().sum() > 53000)], axis=1)
macfail.columns.nunique()


# In[ ]:


# Changing the working directory (changed kaggle to ..)
import os
os.chdir(r'../working')


# In[ ]:


# Writing the modified DF
macfail.to_csv(r'new_df.csv')


# In[ ]:


# Creating a link to access the same
from IPython.display import FileLink
FileLink(r'new_df.csv')


# In[ ]:


# Checking the reading of the new DF
df = pd.read_csv('../working/new_df.csv')
df.head()


# In[ ]:


df = df.drop(['Unnamed: 0'], axis = 1)
df.head()


# In[ ]:


df.columns.nunique()


# ### Not sure why the Unnamed: 0 column is being created. 