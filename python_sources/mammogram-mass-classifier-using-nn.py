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

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier


# In[2]:


col_names=['BI_RADS', 'age', 'shape', 'margin', 'density','severity']
file=pd.read_csv("../input/mammographic_masses.data.txt", names = col_names, na_values='?')
file.head(5)


# In[3]:


file.isnull().sum()


# In[4]:


file.shape


# In[5]:


file.dropna(inplace=True)
file.shape


# In[6]:


X = file.iloc[:,1:-1]
y = file.iloc[:,-1]


# In[7]:


scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled


# In[8]:


def create_model1():
    model = Sequential()
    model.add(Dense(6, input_dim=4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def create_model2():
    model = Sequential()
    model.add(Dense(6, input_dim=4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[ ]:


estimator1a = KerasClassifier(build_fn=create_model1, epochs=100,batch_size = 32, verbose=0)
cv_scores = cross_val_score(estimator1a, X_scaled, y, cv=10)
cv_scores.mean()


# In[ ]:


estimator1b = KerasClassifier(build_fn=create_model1, epochs=100,batch_size = 8, verbose=0)
cv_scores = cross_val_score(estimator1b, X_scaled, y, cv=10)
cv_scores.mean()


# In[9]:


estimator2 = KerasClassifier(build_fn=create_model2, epochs = 100, batch_size = 32, verbose = 0)
cv_scores = cross_val_score(estimator, X_scaled, y, cv=10)
cv_scores.mean()


# In[ ]:


estimator3 = LogisticRegression(solver="liblinear", random_state=0)
cv_scores = cross_val_score(estimator3, X_scaled, y, cv=10)
cv_scores.mean()


# In[ ]:


estimator4 = RandomForestClassifier(max_depth=3, n_estimators=100, random_state=0)
cv_scores = cross_val_score(estimator4, X_scaled, y, cv=10)
cv_scores.mean()

