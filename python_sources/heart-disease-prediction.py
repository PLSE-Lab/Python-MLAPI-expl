#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


import numpy as np
a = np.arange(1,16).reshape((3,5))
b = np.zeros((2,2))
print(a)


# In[4]:


import pandas as pd
from sklearn.utils import shuffle

df = shuffle(pd.read_csv('../input/heart.csv'))
df.head()


# In[5]:


import numpy as np
age = np.array(df['age'])


# In[6]:


import matplotlib.pyplot as plt
n, bins, patches = plt.hist(x=age)


# Correlation between age and target

# In[7]:


df['age'].corr(df['target'])


# In[8]:


df[['oldpeak','target']].head()


# In[9]:


df['oldpeak'].corr(df['target'])


# In[12]:


import matplotlib.pyplot as plt

correlation = []
df_copy = df.copy()
df_copy = df_copy.drop(['target'], axis=1)
for each in df_copy.columns:
    correlation.append(df_copy[each].corr(df['target']))


# In[13]:


plt.plot(correlation)


# In[14]:


related = [correlation.index(each) for each in correlation if each>=0.2]
print(related)


# In[15]:


from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

train, test = train_test_split(df);
print(train.head())
print(test.head())


# In[16]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()


# In[17]:


train_copy = train.copy()
train_x = train_copy.drop(['target'], axis=1)
train_y = train_copy['target']
test_copy = test.copy()
test_x = test_copy.drop(['target'], axis=1)
print(train_x.head())
print(test_x.head())
print(train_y.head())


# In[18]:


model.fit(train_x, train_y)
'''
model.predict(train_x)
model.mean_absolute_error()
'''

