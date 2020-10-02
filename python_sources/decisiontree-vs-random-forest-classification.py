#!/usr/bin/env python
# coding: utf-8

# **Imports**

# In[564]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor # RandomForest
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.7


# **Import Data**

# In[565]:


# Path of the file to read. We changed the directory structure to simplify submitting to a competition
iowa_train_file_path = '../input/train.csv'
iowa_test_file_path = '../input/test.csv'

test_dt = pd.read_csv(iowa_test_file_path)
train_dt = pd.read_csv(iowa_train_file_path)


# **Data Visualization**

# In[566]:


train_dt.head()


# In[567]:


print('Number of train data is: {}' .format(train_dt.shape[0]))


# In[568]:


test_dt.head()


# In[569]:


print('Number of test data: {}' .format(test_dt.shape[0]))


# In[570]:


sns.countplot(x='Survived', data=train_dt);


# In[571]:


sns.countplot(x='Sex', data=train_dt)


# In[572]:


sns.catplot(x='Survived', col='Sex', kind='count', data=train_dt)


# In[573]:


sns.catplot(x='Survived',col='Pclass', kind='count', data=train_dt)


# In[574]:


sns.catplot(x='Survived',col='Embarked', kind='count', data=train_dt)


# **Feature Scaling**

# In[575]:


# Scaling Feature Sex.

m = {'male':1, 'female':0}
train_dt['Sex'] = train_dt['Sex'].map(m)
#train_dt['Sex'] = train_dt['Sex'].str[0].str.lower().map(m)
test_dt['Sex'] = test_dt['Sex'].map(m);


# In[576]:


#Scaling Feature Embarked 

e = {'C': 1, 'Q': 2, 'S': 3}
train_dt['Embarked'] = train_dt['Embarked'].map(e)
test_dt['Embarked'] = test_dt['Embarked'].map(e)


# In[577]:


#Removing Unnecessery Columns from training data
train_dt = train_dt.drop(['Cabin','Name', 'Ticket', 'Fare', 'Parch' ], axis=1)
#train_dt = train_dt.drop('Name', axis=1)
#train_dt = train_dt.drop('Ticket', axis=1)


# In[578]:


#Check training columns
train_dt.head(4)


# In[579]:


#Removing Unnecessery Columns from test data
test_dt = test_dt.drop(['Cabin','Name', 'Ticket', 'Fare', 'Parch' ], axis=1)


# In[580]:


# Check test Columns
test_dt.head(4)


# **Remove Null Values**

# In[581]:


# Calculate number of Null Values
train_dt.isnull().sum()


# In[582]:


#Remove Age and Embarked null values
age_missing = train_dt[train_dt.Age.isnull()].index
train_dt.drop(age_missing, inplace=True)

embarkedMissing = train_dt[train_dt.Embarked.isnull()].index
train_dt.drop(embarkedMissing,inplace=True)


# In[583]:


# Recalculate null values
train_dt.isnull().sum()


# **Creating Extra functions**

# In[584]:


def binarise(preds):
    bin_preds = []
    for p in preds:
        bin_preds.append(step(p))
    return bin_preds 

def step(x, threshold=0.6):
    if x >= threshold:
        return 1
    else: 
        return 0 


# **Model Creation**

# In[585]:


#Split Train and test data
y = train_dt.Survived
#'PassengerId',
features = [ 'Pclass', 'Sex', 'Embarked'] # Sex and Embarked are omited for now
print('data Heads: ', train_dt.columns)


x = train_dt[features] 

print(features)
# Split into validation and training data
train_X, test_X, train_Y, test_Y = train_test_split(x, y, random_state=1)


# In[586]:


train_X.sample(5)


# In[587]:


# Create Model and Predict
dt_model = DecisionTreeClassifier(random_state=1)
dt_model.fit(train_X, train_Y)
pred_dt = dt_model.predict(test_X)


# In[588]:


test_Y.sample(5)


# **Check Accuracy**

# In[589]:


print('Checking accuracy for Decision tree classifier')
print('Mean Absolute error: ' ,mean_absolute_error(test_Y, pred_dt))
print('Mean Squared error: ' ,mean_squared_error(test_Y, pred_dt))
print('Root Mean Squared error: ' ,np.sqrt(mean_squared_error(test_Y,pred_dt)))
print('')
print('Decision Tree Accuracy: {} ' .format(accuracy_score(test_Y,pred_dt)) )


# **Lets Check Accuracy with Random Forest**

# In[590]:


# Create model and predict
rf_model = RandomForestRegressor(n_estimators=20, random_state=0)

rf_model.fit(train_X, train_Y)
pred_rf = rf_model.predict(test_X)


# In[591]:


# Check Accuracy for Random Forest

print('Checking accuracy for Random Forest classifier')
print('Mean Absolute error: ' ,mean_absolute_error(test_Y, pred_rf))
print('Mean Squared error: ' ,mean_squared_error(test_Y, pred_rf))
print('Root Mean Squared error: ' ,np.sqrt(mean_squared_error(test_Y,pred_rf)))
print('')
print('Random Forest Accuracy: {} ' .format(accuracy_score(test_Y , binarise(pred_rf)) ))


# > **File Output**

# **
# we will put random forest's result in output file.**

# In[592]:


#Create test data 
ids = test_dt.PassengerId
test = test_dt[features]

#Predict with test data
preds = rf_model.predict(test.values)
preds = binarise(preds)


# In[593]:


# Write to output file
d = {"PassengerId" : ids.values, "Survived" : preds}
survivors = pd.DataFrame(data=d) 
survivors.to_csv("predictions.csv", index=False)

