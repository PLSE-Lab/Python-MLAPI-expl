#!/usr/bin/env python
# coding: utf-8

# In[ ]:


""" I AM NEW TO THID FIELD, SO PLEASE HELP ME IMPROVE """
#import required libraries
import numpy as np
import pandas as pd


# In[ ]:


#importing the dataset
from subprocess import check_output
dataset = pd.read_csv("../input/HR_comma_sep.csv")


# In[ ]:


#seperate input and required values
X = dataset.iloc[:,[0,1,2,3,4,5,7,8,9] ].values
y = dataset.iloc[:, 6].values


# In[ ]:


# encode the sales and salary since they are strings
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X  = LabelEncoder()
X[:, 7] = labelencoder_X.fit_transform(X[:, 7])
X[:, 8] = labelencoder_X.fit_transform(X[:, 8])
onehotencoder = OneHotEncoder(categorical_features = [7, 8]) #onehotencode them since they don't have a particular value
X = onehotencoder.fit_transform(X).toarray() # fit, transform and change to array


# In[ ]:


#split the data
"""Ignore the depreciation warning"""
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[ ]:


#fit SVM with 'rbf' kernel
from sklearn.svm import SVC
clf = SVC(kernel = 'rbf', random_state = 0)
clf.fit(X_train, y_train)


# In[ ]:


#predict the y_test from X_test
y_pred = clf.predict(X_test)


# In[2]:


#see the accuracy
from sklearn.metrics import accuracy_score
accuracy_svm = accuracy_score(y_test, y_pred, normalize = True)
print(accuracy_svm)

