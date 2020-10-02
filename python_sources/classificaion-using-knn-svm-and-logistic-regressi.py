#!/usr/bin/env python
# coding: utf-8

# In[76]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn import preprocessing, neighbors
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[77]:


df = pd.read_csv('../input/heart.csv')
print(df.head())
df.describe()
df.info()


# 

# Here some columns aer renamed, so they are easy to understand.
# 
# Also some columns are filtered out from the data frame. 

# In[78]:


df.rename(columns={'cp': 'chest_pain_type', 'trestbps' :'blood_pressure', 'chol':'cholesterol', 'fbs':'blood_sugar_rest', 'thalach':'max_heart_rate' }, inplace=True)
df = df[['age', 'sex', 'chest_pain_type', 'blood_pressure', 'cholesterol', 'blood_sugar_rest', 'max_heart_rate', 'target']  ]
df.head(5)


# We indetify the inputs in X and the output in y. Output is the target Column. The goal is to identiify whether a patient has heart disease or not. 

# In[79]:


X = np.array(df.drop(['target'], 1))
y = np.array(df['target'])


# Train test Split. Standard proceduer of training with 80% of the data and Keeping the 20% for testing.

# In[80]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# KNN Classification

# In[81]:


clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
accuracy_KNN = clf.score(X_test, y_test)
print(accuracy_KNN)


# Support Vector Machine Classification

# In[82]:


from sklearn import svm

clf = svm.SVC(gamma='scale')

clf.fit(X_train, y_train)
accuracy_SVM = clf.score(X_test, y_test)
print(accuracy_SVM)


# Logistic Regression

# In[83]:


from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(solver = 'newton-cg')

clf.fit(X_train, y_train)
accuracy_LR = clf.score(X_test, y_test)
print(accuracy_LR)


# From the results it can be seen that KNN and SVM usually have accuracy ranging from 60%-70%.
# While ogistic regression have accuracy of 75%-85%.
# 
# For this data Logistic Regression seem to be better suited. 

# In[ ]:




