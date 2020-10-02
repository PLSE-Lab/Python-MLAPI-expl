#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from pandas.plotting import scatter_matrix
import seaborn as sns # data visualization library  

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


train=pd.read_csv('/home/harish/Downloads/train.csv')
test=pd.read_csv('/home/harish/Downloads/test.csv')
PassengerId = test['PassengerId']


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


full_data = [train, test]
for dataset in full_data:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train['Title'], train['Sex'])


# In[ ]:


for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[ ]:


for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
for dataset in full_data:
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
test['Fare'].fillna(test['Fare'].dropna().median(), inplace=True)

for dataset in full_data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

for dataset in full_data:
    dataset.loc[ dataset['Age'] <= 16, 'Age']    = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4 ;
    
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    
title_mapping = {"Mr": 4, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 5}
for dataset in full_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

for dataset in full_data:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass


train.head()
    


# In[ ]:


test.head()


# In[ ]:


drop_list1 = ['Cabin','Ticket','PassengerId','Name','SibSp','Parch','FamilySize']
train_1=train.drop(drop_list1,axis=1)
test_1=test.drop(drop_list1,axis=1)
train_1.head()


# In[ ]:


test_1.head()


# In[ ]:





# In[ ]:


f,ax = plt.subplots(figsize=(15,15))
sns.heatmap(train_1.corr(), annot=True, linewidths=.6, fmt= '.2f',ax=ax)


# In[ ]:


train_1.hist(figsize=(10,10))
plt.show()


# In[ ]:


train_1.describe()


# In[ ]:


test_1.describe()


# In[ ]:


test_1['Pclass'].fillna(test_1['Pclass'].dropna().median(), inplace=True)
test_1['Sex'].fillna(test_1['Sex'].dropna().median(), inplace=True)
test_1['Age'].fillna(test_1['Age'].dropna().median(), inplace=True)
test_1['Fare'].fillna(test_1['Fare'].dropna().median(), inplace=True)
test_1['Embarked'].fillna(test_1['Embarked'].dropna().median(), inplace=True)
test_1['Title'].fillna(test_1['Title'].dropna().median(), inplace=True)
test_1['IsAlone'].fillna(test_1['IsAlone'].dropna().median(), inplace=True)
test_1['Age*Class'].fillna(test_1['Age*Class'].dropna().median(), inplace=True)


# In[ ]:


train_1['Pclass'].fillna(train_1['Pclass'].dropna().median(), inplace=True)
train_1['Sex'].fillna(train_1['Sex'].dropna().median(), inplace=True)
train_1['Age'].fillna(train_1['Age'].dropna().median(), inplace=True)
train_1['Fare'].fillna(train_1['Fare'].dropna().median(), inplace=True)
train_1['Embarked'].fillna(train_1['Embarked'].dropna().median(), inplace=True)
train_1['Title'].fillna(train_1['Title'].dropna().median(), inplace=True)
train_1['IsAlone'].fillna(train_1['IsAlone'].dropna().median(), inplace=True)
train_1['Age*Class'].fillna(train_1['Age*Class'].dropna().median(), inplace=True)


# In[ ]:


X_train = train_1.drop("Survived", axis=1)
Y_train = train_1["Survived"]
X_test  = test_1
X_train.shape, Y_train.shape, X_test.shape


# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 4)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn


# In[ ]:


# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc


# In[ ]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree


# In[ ]:


# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": PassengerId ,
        "Survived": Y_pred
    })
submission.to_csv('/home/harish/Downloads/submission1.csv', index=False)


# In[ ]:




