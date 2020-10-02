#!/usr/bin/env python
# coding: utf-8

# In[94]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[95]:


# save filepath to variable
genderPath = '../input/gender_submission.csv'
trainPath = '../input/train.csv'
testPath = '../input/test.csv'

# read data and store in dataframe
genderData = pd.read_csv(genderPath)
trainDataNotImputer = pd.read_csv(trainPath)
testDataNotImputer = pd.read_csv(testPath)


# In[96]:


# XTestHotEnc = pd.get_dummies(XTest)
# XTestHotEnc.head()

trainDataHotKey = pd.get_dummies(trainDataNotImputer)
testDataHotKey = pd.get_dummies(testDataNotImputer)


# In[104]:


# trainData = trainData.dropna(axis=0)
from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()
trainData = my_imputer.fit_transform(trainDataHotKey)
testData = my_imputer.fit_transform(testDataHotKey)


# In[101]:


trainData


# In[ ]:


features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']
X = trainData[features]
XTest = testData[features]


# In[ ]:


# describe train data
X.describe()


# In[ ]:


X.head()


# In[ ]:


# XOneHotEnc = pd.get_dummies(X)
# XOneHotEnc.head()


# In[ ]:


from sklearn.tree import DecisionTreeClassifier

#Create classifier object with default hyperparameters
model = DecisionTreeClassifier()  

#Fit our classifier using the training features and the training target values
model.fit(XOneHotEnc, y)


# In[ ]:


from sklearn.metrics import mean_absolute_error

prediction = model.predict(XOneHotEnc)
print(mean_absolute_error(y, prediction))


# In[ ]:


from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y =  train_test_split(XOneHotEnc, y, random_state = 0)
model.fit(train_X,train_y)

val_predictions = model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))


# In[ ]:


XTest.describe()


# In[ ]:


# XTestHotEnc = pd.get_dummies(XTest)
# XTestHotEnc.head()


# In[ ]:


testPrediction = model.predict(XTestHotEnc)


# In[ ]:



# make submission


submission = pd.DataFrame({'PassengerId':testData.PassengerId,'Survived':testPrediction})
submission.head()


# In[ ]:


filename = 'Titanic Predictions 1.csv'

submission.to_csv(filename,index=False)


# In[ ]:




