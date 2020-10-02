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


train = pd.read_csv('../input/Admission_Predict.csv')


# In[3]:


train.head()


# In[4]:


columns = train.columns


# In[5]:


x_train = train[columns[1:-1]]
y_train = train[columns[-1]]


# In[6]:


max_ = []
for col in x_train.columns:
    max_.append(np.max(x_train[col]))
    x_train[col] = x_train[col].divide(np.max(x_train[col]))


# In[7]:


x_train.head()


# In[21]:


y_train.head()


# In[8]:


from keras.layers import *
from keras.models import *


# In[35]:


model = Sequential()

model.add(Dense(1, activation='relu', input_shape=(7,)))

model.compile('adam', 'mean_squared_error')
model.summary()
model.fit(x_train, y_train, 32, 10)


# In[36]:


model.predict(x_train[:5])

