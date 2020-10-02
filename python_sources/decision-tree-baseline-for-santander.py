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


import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.preprocessing import StandardScaler


# In[3]:


# Read traning dataset
train = pd.read_csv("../input/train.csv")
train.head()


# In[4]:


X_train = train.iloc[:,2:].values
y_train = train.iloc[:,1].values
scaler = StandardScaler()
x_train = scaler.fit_transform(X_train)

print("x_train shape:", x_train.shape, "y_train:", y_train.shape)
x_train[:3]


# In[5]:


xTrain, xTest, yTrain, yTest = train_test_split(x_train, y_train, test_size = 0.10, random_state = 2019)
print("xTrain shape:", xTrain.shape, "yTrain:", yTrain.shape, "xTest shape:", xTest.shape, "yTest shape: ", yTest.shape)


# In[6]:


# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(xTrain, yTrain)


# In[7]:


#Predict the response for test dataset
y_pred = clf.predict(xTest)


# In[12]:


y_pred[:3]


# In[8]:


# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(yTest, y_pred))


# In[9]:


# Read test dataset
test = pd.read_csv('../input/test.csv')


# In[13]:


X_test = test.iloc[:,1:].values
x_test = scaler.fit_transform(X_test)
print(x_test.shape)
x_test[:3]


# In[14]:


predictions = clf.predict(x_test)


# In[18]:


ans = pd.DataFrame(predictions)
ans = ans[0]
test_id = test["ID_code"]
sub = pd.DataFrame()
sub['ID_code'] = test_id
sub['target'] = ans


# In[ ]:


sub.to_csv('submission_dt1.csv', index=False)

