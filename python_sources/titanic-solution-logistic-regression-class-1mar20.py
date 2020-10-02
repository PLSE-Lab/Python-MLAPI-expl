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


train = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")


# In[ ]:


train.head()


# In[ ]:


test.info()


# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


#Treat Age
train['Age'] = train['Age'].fillna(value=train['Age'].median())
test['Age'] = test['Age'].fillna(value=test['Age'].median())


# In[ ]:


#Treat Fare
train['Fare'] = train['Fare'].fillna(value=train['Fare'].median())
test['Fare'] = test['Fare'].fillna(value=test['Fare'].median())


# In[ ]:


train['Embarked'].mode()[0]


# In[ ]:


#Treat Embarked
train['Embarked'] = train['Embarked'].fillna(value=train['Embarked'].mode()[0])
test['Embarked'] = test['Embarked'].fillna(value=test['Embarked'].mode()[0])


# In[ ]:


#Treat Cabin
train['Cabin'] = train['Cabin'].fillna('Missing')
train['Cabin'] = train['Cabin'].str[0]
train['Cabin'].value_counts()


# In[ ]:


#Treat Cabin
test['Cabin'] = test['Cabin'].fillna('Missing')
test['Cabin'] = test['Cabin'].str[0]
test['Cabin'].value_counts()


# In[ ]:


#Extract Title from Name
train['Title'] = train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
test['Title'] = test['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)


# In[ ]:


#We will combine a few categories, since few of them are unique 
train['Title'] = train['Title'].replace(['Capt', 'Dr', 'Major', 'Rev'], 'Officer')
train['Title'] = train['Title'].replace(['Lady', 'Countess', 'Don', 'Sir', 'Jonkheer', 'Dona'], 'Royal')
train['Title'] = train['Title'].replace(['Mlle', 'Ms'], 'Miss')
train['Title'] = train['Title'].replace(['Mme'], 'Mrs')
train['Title'].value_counts()


# In[ ]:


#We will combine a few categories, since few of them are unique 
test['Title'] = test['Title'].replace(['Capt', 'Dr', 'Major', 'Rev'], 'Officer')
test['Title'] = test['Title'].replace(['Lady', 'Countess', 'Don', 'Sir', 'Jonkheer', 'Dona'], 'Royal')
test['Title'] = test['Title'].replace(['Mlle', 'Ms'], 'Miss')
test['Title'] = test['Title'].replace(['Mme'], 'Mrs')
test['Title'].value_counts()


# In[ ]:


#Family Size & Alone 
train['Family_Size'] = train['SibSp'] + train['Parch'] + 1
train['IsAlone'] = 0
train.loc[train['Family_Size']==1, 'IsAlone'] = 1


# In[ ]:


#Family Size & Alone 
test['Family_Size'] = test['SibSp'] + test['Parch'] + 1
test['IsAlone'] = 0
test.loc[train['Family_Size']==1, 'IsAlone'] = 1


# In[ ]:


train.head()


# # Process Data for Modelling

# In[ ]:


all = pd.concat([train, test], sort = False)
all.info()


# In[ ]:


all_dummies = pd.get_dummies(all.drop(['Name', 'Ticket'], axis = 1), drop_first = True)
all_dummies.head()


# In[ ]:


all_train = all_dummies[all_dummies['Survived'].notna()]


# In[ ]:


all_test = all_dummies[all_dummies['Survived'].isna()]


# # Log Model

# In[ ]:


y = all_train['Survived']
X = all_train.drop(['Survived', 'PassengerId'], axis = 1)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30, random_state=101)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logModel = LogisticRegression(max_iter = 5000)
logModel.fit(X_train, y_train)


# In[ ]:


predictions = logModel.predict(X_test)
predictions


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))


# In[ ]:


logModel.score(X_train, y_train)


# In[ ]:


logModel.score(X_test, y_test)


# # Make Submission

# In[ ]:


X_Submission = all_test.drop(['PassengerId', 'Survived'], axis = 1)


# In[ ]:


pred_for_submission = logModel.predict(X_Submission).astype(int)


# In[ ]:


logSub = pd.DataFrame({'PassengerId': all_test['PassengerId'], 'Survived':pred_for_submission })
logSub.head(1)


# In[ ]:


logSub.to_csv("1_Logistics_Regression_Submission.csv", index = False)

