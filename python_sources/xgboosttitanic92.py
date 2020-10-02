#!/usr/bin/env python
# coding: utf-8

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





# In[ ]:


import pandas as pd
gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")
test = pd.read_csv("../input/titanic/test.csv")
train = pd.read_csv("../input/titanic/train.csv")


# In[ ]:


test.describe()


# In[ ]:


train.describe()


# In[ ]:


train.columns.values


# In[ ]:


test.columns.values


# In[ ]:


train.shape


# In[ ]:


train.head(2)


# In[ ]:


test.head(2)


# In[ ]:


#locate missing values within the train columns
percent_null = train.isnull().sum()/train.isnull().count()
percent_null.sort_values(ascending=False)
#we will drop the Cabin featue later due to its incompleteness


# In[ ]:


#locate missing values within the test columns
percent_null = test.isnull().sum()/test.isnull().count()
percent_null.sort_values(ascending=False)


# In[ ]:


train['Embarked'].value_counts()


# In[ ]:


train['Age'] = train['Age'].fillna(train.mean)


# In[ ]:


test.describe()
#some of the age and fare rates are missing.


# In[ ]:


#locate missing values within the features
percent_null_test = test.isnull().sum()/test.isnull().count()
percent_null_test.sort_values(ascending=False)


# In[ ]:


test.shape


# In[ ]:


#Survival rate
#1 denotes a survived passenger
train.groupby('Pclass').Survived.value_counts()


# In[ ]:


survived = train[train['Survived'] == 1]
not_survived = train[train['Survived'] == 0]

print ("Survived: %i (%.1f%%)"%(len(survived), float(len(survived)) /len(train)*100.0))
print ("Not Survived: %i (%.1f%%)"%(len(not_survived), float(len(not_survived))/len(train)*100.0))
print ("Total: %i"%len(train))


# In[ ]:


train [['Survived','Pclass']].groupby(['Pclass'], as_index=False).mean()


# In[ ]:


train.corr()["Survived"].sort_values(ascending=False)


# In[ ]:


#drop any column within +-.1 of 0
drop_columns = ['PassengerId','Parch','SibSp','Ticket','Cabin']
for column in drop_columns:
    train = train.drop(column, axis=1)


# In[ ]:


#look at data after we remove unneeded features
train.describe()


# In[ ]:


train.head()


# In[ ]:


#creates a columns of the Title
train_test_data = [train, test] # combining train and test dataset

for dataset in train_test_data:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.')


# In[ ]:


train.head()


# In[ ]:


for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',  	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[ ]:


# make it numeric
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5}
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)


# In[ ]:


for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)


# In[ ]:


#check our progress
train.tail()


# In[ ]:


for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')


# In[ ]:


for dataset in train_test_data:
    #print(dataset.Embarked.unique())
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)


# In[ ]:


train.columns.values


# In[ ]:


#locate missing values within the features
percent_null = train.isnull().sum()/train.isnull().count()
percent_null.sort_values(ascending=False)
#we will drop the Cabin featue later due to its incompleteness


# In[ ]:


train.Embarked.value_counts()
# notice there is not any more nan values


# In[ ]:


test.columns.values


# In[ ]:


#let drop more festures
drop_train = ['Name','Age']
for column in drop_train:
    train = train.drop(column, axis=1)
drop_test = ['Cabin','Name','Ticket','Age','Parch','SibSp']
for column in drop_test:
    test = test.drop(column,axis=1)


# In[ ]:


test.head()


# In[ ]:


#make sure columns align between train and test data sets
train.columns.values


# In[ ]:


test.columns.values


# In[ ]:


#need to concat the people_dummies with the numerical values


# In[ ]:


test=test.fillna(test.mean())


# In[ ]:





# Model Training

# In[ ]:


X_train = train.drop('Survived', axis=1)
y_train = train['Survived']
X_test = test.drop("PassengerId", axis=1).copy()

X_train.shape, y_train.shape, X_test.shape


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


# In[ ]:


clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred_log_reg = clf.predict(X_test)
acc_log_reg = round( clf.score(X_train, y_train) * 100, 2)
print (str(acc_log_reg) + ' percent')


# In[ ]:


clf = SVC()
clf.fit(X_train, y_train)
y_pred_svc = clf.predict(X_test)
acc_svc = round(clf.score(X_train, y_train)*100, 3)
print(str(acc_svc) + ' percent')


# In[ ]:


#decision tree
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred_decision_tree = clf.predict(X_test)
acc_decision = round(clf.score(X_train, y_train)*100, 2)
print (str(acc_decision)+ ' percent')


# In[ ]:


y_pred_decision_tree = clf.predict(X_test)


# In[ ]:


y_pred_decision_tree


# In[ ]:


submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':y_pred_decision_tree})


# In[ ]:


submission.head()


# In[ ]:


filename = 'Titanic Predictions 2.csv'

submission.to_csv(filename,index=False)
print('Saved File: ' + filename)


# In[ ]:


#incorporate xgboost
#xg boost here
clf = XGBClassifier()
clf.fit(X_train, y_train)
y_pred_XGBClassifier = clf.predict(X_test)
acc_XGB = round(clf.score(X_train, y_train)*100, 2)
print(str(acc_XGB)+ ' percent')


# In[ ]:


y_pred_XGBClassifier


# In[ ]:


y_pred_decision_tree = clf.predict(X_test)


# In[ ]:


submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':y_pred_decision_tree})


# In[ ]:


filename = 'Titanic Predictions xgb.csv'

submission.to_csv(filename,index=False)
print('Saved File: ' + filename)

