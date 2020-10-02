#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')


# In[ ]:


train.shape,test.shape


# In[ ]:


train.head()


# In[ ]:


train.describe(include='all')


# In[ ]:


train.isnull().sum()


# In[ ]:


sns.barplot(x='Sex',y='Survived',data=train)


# In[ ]:


sns.barplot(x='Pclass',y='Survived',data=train)


# In[ ]:


train=train.drop(['Cabin'],axis=1)
test=test.drop(['Cabin'],axis=1)


# In[ ]:


train=train.drop(['Ticket'],axis=1)
test=test.drop(['Ticket'],axis=1)


# In[ ]:


train['Embarked'].value_counts()


# In[ ]:


train['Embarked'].fillna('S',inplace=True)


# In[ ]:


combine=[train,test]
for dataset in combine:
    dataset['Title']=dataset.Name.str.extract('([A-Za-z]+)\.',expand=False)
pd.crosstab(train['Title'],train['Sex'])    


# In[ ]:


for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col',
    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona','Countess','Lady','Sir'], 'Rare')
    dataset['Title']=dataset['Title'].replace('Mlle','Miss')
    dataset['Title']=dataset['Title'].replace('Ms','Miss')
    dataset['Title']=dataset['Title'].replace('Mme','Mrs')
train[['Title','Survived']].groupby(['Title'],as_index=False).mean() 


# In[ ]:


title_mapping={'Mr':1,"Miss":2,'Mrs':3,'Master':4,'Rare':5}
for dataset in combine:
    dataset['Title']=dataset['Title'].map(title_mapping)
    dataset['Title']=dataset['Title'].fillna(0)


# In[ ]:


train.head()


# In[ ]:


train['Age'].groupby(train['Title']).mean()


# In[ ]:


age_title_mapping = {1: 33, 2: 22, 3: 36, 4: 5, 5: 46}
train = train.fillna({"Age": train["Title"].map(age_title_mapping)})
test = test.fillna({"Age": test["Title"].map(age_title_mapping)})


# In[ ]:


train['Age_band']=0
train.loc[train['Age']<=16,'Age_band']=0
train.loc[(train['Age']>16)&(train['Age']<=32),'Age_band']=1
train.loc[(train['Age']>32)&(train['Age']<=48),'Age_band']=2
train.loc[(train['Age']>48)&(train['Age']<=64),'Age_band']=3
train.loc[train['Age']>64,'Age_band']=4
train.head(2)


# In[ ]:


test['Age_band']=0
test.loc[test['Age']<=16,'Age_band']=0
test.loc[(test['Age']>16)&(test['Age']<=32),'Age_band']=1
test.loc[(test['Age']>32)&(test['Age']<=48),'Age_band']=2
test.loc[(test['Age']>48)&(test['Age']<=64),'Age_band']=3
test.loc[test['Age']>64,'Age_band']=4
test.head(2)


# In[ ]:


train = train.drop(['Age'], axis = 1)
test = test.drop(['Age'], axis = 1)


# In[ ]:


train = train.drop(['Name'], axis = 1)
test = test.drop(['Name'], axis = 1)


# In[ ]:


sex_mapping = {"male": 0, "female": 1}
train['Sex'] = train['Sex'].map(sex_mapping)
test['Sex'] = test['Sex'].map(sex_mapping)

train.head()


# In[ ]:


embarked_mapping = {"S": 1, "C": 2, "Q": 3}
train['Embarked'] = train['Embarked'].map(embarked_mapping)
test['Embarked'] = test['Embarked'].map(embarked_mapping)

train.head()


# In[ ]:


test.isnull().sum()


# In[ ]:


test['Fare'].fillna(test['Fare'].dropna().median(), inplace=True)
test.head()


# In[ ]:


train['FareBand'] = pd.qcut(train['Fare'], 4, labels = [1, 2, 3, 4])
test['FareBand'] = pd.qcut(test['Fare'], 4, labels = [1, 2, 3, 4])
train = train.drop(['Fare'], axis = 1)
test = test.drop(['Fare'], axis = 1)


# In[ ]:


from sklearn.model_selection import train_test_split

x = train.drop(['Survived', 'PassengerId'], axis=1)
y = train["Survived"]
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.2, random_state = 0)


# In[ ]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_val)
acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gaussian)


# In[ ]:


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_logreg)


# In[ ]:


from sklearn.svm import SVC

svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_val)
acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_svc)


# In[ ]:


from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_val)
acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_linear_svc)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_decisiontree)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_randomforest)


# In[ ]:


ids = test['PassengerId']
predictions = randomforest.predict(test.drop('PassengerId', axis=1))


output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('submission.csv', index=False)


# In[ ]:




