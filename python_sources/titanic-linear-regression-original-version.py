#!/usr/bin/env python
# coding: utf-8

# (See [Version 2](https://www.kaggle.com/niteshhalai/titanic-linear-regression-v2) for a model with slightly better accuracy.

# I am trying to use linear regression to the titanic dataset to see if it results in any meaningful prediction

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('/kaggle/input/titanic/train.csv')
train.head(10)


# In[ ]:


train.info()


# First I will  split the data between variable and targets. I will also remove the columns PassenderId, Name, Ticket (these should not have any effect on whether the passenger survived or not) and Cabin (as these have a lot of missing values). I will fill the 

# In[ ]:


y = train['Survived']
train.drop(labels = ['Survived','PassengerId','Name','Ticket','Cabin'], axis = 1, inplace = True)


# In[ ]:


train.head()


# In[ ]:


train.info()


# Filling the age missing age values with mean age.

# In[ ]:


train['Age'].fillna(train['Age'].mean(), inplace = True)


# In[ ]:


train.info()


# I want to turn the sex and Embarked columns to dummy variables using one hot encoding:

# In[ ]:


categorical_columns = ['Sex','Embarked']
train = pd.get_dummies(train,columns = categorical_columns, dtype = int)


# In[ ]:


train.info()


# In[ ]:


X = []
for column in train.columns:
    X.append(column)
X = train[X]
X


# In[ ]:


from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X, y)


# Making predictions for train data set:

# In[ ]:


y_pred = model.predict(train)
y_pred


# Getting the range of the predictions

# In[ ]:


print(y_pred.max())
print(y_pred.min())


# Function to convert the prediction to a one (survived) or zer (not survived)

# In[ ]:


def one_or_zero(abc):
    if (1 - abc) < (abc - 0):
        return 1
    else: 
        return 0


# Converting the predictions to 1 or 0:

# In[ ]:


list_of_predictions = []

for pred in y_pred:
    list_of_predictions.append(one_or_zero(pred))
    
y_pred = np.asarray(list_of_predictions)
y_pred


# Accuracy of the model on the same data it trained on:

# In[ ]:


unique, counts = np.unique( np.asarray(y_pred == y), return_counts=True)
true_false_values = dict(zip(unique, counts))
accuracy = true_false_values[True]/len(np.asarray(y_pred == y))
accuracy


# **Using the model on the test data**

# In[ ]:


original_test = pd.read_csv('/kaggle/input/titanic/test.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
test.drop(labels = ['PassengerId','Name','Ticket','Cabin'], axis = 1, inplace = True)
test['Age'].fillna(test['Age'].mean(), inplace = True)
categorical_columns = ['Sex','Embarked']
test = pd.get_dummies(test,columns = categorical_columns, dtype = int)


# Looks like there are some null values in the test data which were not there in train:

# In[ ]:


test.info()


# Filling the null value in Fare column with the mean:

# In[ ]:


test['Fare'].fillna(test['Fare'].mean(), inplace = True)


# In[ ]:


test.info()


# In[ ]:


test_pred = model.predict(test)
list_of_predictions_test = []

for pred in test_pred:
    list_of_predictions_test.append(one_or_zero(pred))
    
test_pred = np.asarray(list_of_predictions_test)
test_pred


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": original_test["PassengerId"],
        "Survived": test_pred
    }) 

filename = 'submission.csv'
submission.to_csv(filename,index=False)
print('Saved file: ' + filename)

