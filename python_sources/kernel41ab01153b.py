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


df=pd.read_csv("/kaggle/input/santa-workshop-tour-2019/family_data.csv")


# In[ ]:


df


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


df.columns


# In[ ]:


df.dtypes


# In[ ]:


df.isna().sum()


# In[ ]:


df.describe()


# In[ ]:


df.corr()


# In[ ]:


df.hist()


# In[ ]:


import seaborn as sns
df.boxplot()


# In[ ]:


sns.boxplot(df["choice_0"])


# In[ ]:


len(df)
df.shape


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




