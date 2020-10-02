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
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import xgboost as xg
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../input/diabetes.csv")


# In[ ]:


df.head()


# In[ ]:


df.describe()


# ### Preprocessing the data
# *  standardising the data 

# In[ ]:


norm_df = StandardScaler()
norm_df.fit(df)


# In[ ]:


df.head()


# * Defining features and target values for the model 

# In[ ]:


X = df.loc[:,'Pregnancies':'Age']
y= df['Outcome']


# In[ ]:


fig  = plt.figure(figsize=(12,20))
X.hist(bins=50 , figsize=(12,8))
plt.show()


# In[ ]:


plt.bar(y , height=6)


# In[ ]:




