#!/usr/bin/env python
# coding: utf-8

# In[15]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras import models
from keras import layers


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


data = pd.read_csv('../input/winequality-red.csv')


# In[3]:


data.head()


# ### Basic stats about the data

# In[4]:


data.describe()


# ### Checking if it has some null values

# In[5]:


data.isnull().sum()


# ### Plotting a scatter matrix to viszulize each pair of features (to find corelation)

# In[8]:


relation_image=sns.pairplot(data)


# In[11]:


#Plotting the correlation values to get better sense of it!
cor = data[data.columns].corr()
correl_image=sns.heatmap(cor,annot=True)


# ### Splitting the data into training and testing set

# In[12]:


X_data = data.drop(['quality'],axis=1)
y_label = data['quality']


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X_data, y_label, test_size=0.3, random_state=42)


# ### Use Keras to make regressor model

# In[19]:


model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(11,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1))

model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

model.fit(X_train, y_train, batch_size=8, epochs=20)


# ### Evaluating the model on the test data

# In[24]:


test_mse, test_mae = model.evaluate(X_test, y_test, verbose=0)
print('Mean Squared Error: ',test_mse, '\n','Mean Absolute Error: ',test_mae)


# In[ ]:




