#!/usr/bin/env python
# coding: utf-8

# At this step, just a few experiments
# inspired from this kernel
# https://www.kaggle.com/poonaml/titanic-survival-prediction-end-to-end-ml-pipeline
# https://en.wikipedia.org/wiki/Passengers_of_the_RMS_Titanic

# Import libraries
# =======

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Import ML Libraries
from sklearn import datasets  
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split  
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import AdaBoostClassifier

from xgboost.sklearn import XGBClassifier  
from xgboost.sklearn import XGBRegressor

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# Read files and assign to pandas dataframes
# ===============================================

# In[ ]:


train = pd.read_csv("../input/train.csv")

# Splits dataset so that we have a separate y vector for results
train_y = train["Survived"]
train_X = train.drop("Survived", axis=1)

test_X = pd.read_csv("../input/test.csv")

# For data quality checks
whole_X = pd.concat([train_X, test_X])


# Prepare data
# ============

# Process Null Values
# -------------------

# In[ ]:


#First, we perform a basic count
df1 = pd.DataFrame(train_X.isnull().sum(), columns=['Null in train'])
df2 = pd.DataFrame(test_X.isnull().sum(), columns=['Null in test'])

df = df1.join(df2)


# In[ ]:


#Let's check in detail how we could fix "Embarked" for 2 lines
whole_X[whole_X['Embarked'].isnull()]

train[train['Embarked'].isnull()]


# In[ ]:


#whole_X[whole_X['Name'].str.contains('Taussig')]
whole_X[whole_X['Cabin'].fillna('').str.startswith('B')].sort_values('Ticket')

whole_X[['Ticket', 'Embarked']].groupby(['Ticket', 'Embarked']).size()


# In[ ]:


whole_X.loc[whole_X['Ticket'] == 'PC 17483']


# one hot, null values, feature engineering

# Create train and test subsets
# ===========

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.3, random_state=42)


# XGBoost
# =======

# In[ ]:


xclas = XGBClassifier() 

# Note: we have to adapt this step so that we take these varaibales into account
xclas.fit(X_train.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1), y_train) 
y_pred = xclas.predict(X_test.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1))

#Evaluate
scores = cross_val_score(xclas, X_train.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1), y_train) 
print(scores.mean())


# In[ ]:


confusion_matrix(y_test, y_pred)


# AdaBoost
# =======

# In[ ]:


adb=AdaBoostClassifier()
adb.fit(X_train.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1),y_train)
cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=50)
scores = cross_val_score(adb, X_train.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1), y_train["Survived"], scoring='f1',cv=cv)
print(scores.mean())

