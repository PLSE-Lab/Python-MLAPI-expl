#!/usr/bin/env python
# coding: utf-8

# **Predicting the Species using Logistic Regression. This is my first kernel on Kaggle.**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[ ]:


#Reading the input csv to dataframe
df = pd.read_csv("../input/Iris.csv")


# In[ ]:


#printing first few records of the data. Here SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm are the feature columns and the Species is the target column which we have to predict
df.head()


# In[ ]:


# Shape of the data. input data has 150 data points 
df.shape


# In[ ]:


#input data has 3 different Species. Each species has 50 data points
df.Species.value_counts()


# In[ ]:


#Splitting the data into train and test sets
train, test = train_test_split(df, test_size=0.2, random_state=0)
features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
X_train = train[features]
y_train = train['Species']

X_test = test[features]
y_test = test['Species']


# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# In[ ]:


#Creating the model.
cls = LogisticRegression().fit(X_train, y_train)
cls


# In[ ]:


#Score on train set
cls.score(X_train, y_train)


# In[ ]:


#Score on test set
cls.score(X_test, y_test)


# In[ ]:


# Score from the above model was good, but lets try with a different regularisation parameter to see if we can improve the model
cls = LogisticRegression(C=6).fit(X_train, y_train) # Default C value is 1. Here we are using 6
cls


# In[ ]:


#Score on train set
cls.score(X_train, y_train)


# In[ ]:


#Score on test set
cls.score(X_test, y_test)


# In[ ]:


#Score improved with a high C Value.

