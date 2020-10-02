#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import tensorflow as tf
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from tensorflow import keras
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/train.csv')
df.head()


# Lets regularize the data

# In[ ]:


X = df.drop(['label'], axis = 1)/255.
y = df['label']


# Let us start the classification with different methods and then use the voting classifier to use them

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = .2, random_state = 0)


# In[ ]:


#RandomForestClassifier
clf1 = RandomForestClassifier(n_estimators = 250)
clf1.fit(X_train, y_train)
print( clf1.score(X_test, y_test))


# In[ ]:


#NeuralNetworks
clf2 = MLPClassifier(hidden_layer_sizes = (400, 400, 400))
clf2.fit(X_train, y_train)
clf2.score(X_test, y_test)


# In[ ]:


#GradientBoostClassifier
clf3 = GradientBoostingClassifier(n_estimators = 200)
clf3.fit(X_train, y_train)
clf3.score(X_test, y_test)


# In[ ]:


#Voting
clf4 = VotingClassifier([('rf', clf1), ('mpl', clf2), ('gbc', clf3)], voting = 'soft')
clf4.fit(X_train, y_train)
clf4.score(X_test, y_test)


# In[ ]:




