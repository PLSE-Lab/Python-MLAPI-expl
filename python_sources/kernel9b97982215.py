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


df = pd.read_csv('../input/heart.csv')


# In[ ]:


df


# In[ ]:


df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']


# In[ ]:


import seaborn as sns


# In[ ]:


sns.distplot(df['sex'])


# In[ ]:


sns.distplot(df['age'])


# In[ ]:


sns.distplot(df['cp'])


# In[ ]:


sns.distplot(df['trestbps'])


# In[ ]:


sns.distplot(df['chol'])


# In[ ]:


sns.distplot(df['fbs'])


# In[ ]:


sns.distplot(df['restecg'])


# In[ ]:


sns.distplot(df['thalach'])


# In[ ]:


sns.distplot(df['exang'])


# In[ ]:


sns.distplot(df['oldpeak'])


# In[ ]:


sns.distplot(df['slope'])


# In[ ]:


sns.distplot(df['ca'])


# In[ ]:


sns.distplot(df['thal'])


# In[ ]:


sns.distplot(df['target'])


# In[ ]:





# In[ ]:




