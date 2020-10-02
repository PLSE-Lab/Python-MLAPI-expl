#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


us_supreme_court_data = "../input/supreme-court/database.csv"
df = pd.read_csv(us_supreme_court_data)


# In[ ]:


df.head()


# In[ ]:


indian_prison = "/kaggle/input/prison-in-india/Caste.csv"
df = pd.read_csv(indian_prison)
df.head()


# In[ ]:


df['caste'].value_counts()


# In[ ]:


df.shape


# ## 

# In[ ]:


us_supreme_court_data = "../input/supreme-court/database.csv"
df = pd.read_csv(us_supreme_court_data)
df.head()


# In[ ]:


df.shape


# In[ ]:


df.corr()


# In[ ]:


df.columns


# In[ ]:


df['minority_votes'].value_counts()


# In[ ]:


plt.plot(df['decision_type'])


# In[ ]:


plt.plot(df['majority_votes'])


# In[ ]:




