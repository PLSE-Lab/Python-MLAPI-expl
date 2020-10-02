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


# 

# In[ ]:


train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


train[train['Embarked'].isnull()]


# In[ ]:


train.drop([61, 829], axis=0,inplace=True)


# In[ ]:


test[test['Fare'].isnull()]


# In[ ]:


test.drop([152], axis=0,inplace=True)


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train,palette='RdBu_r')


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')


# In[ ]:


sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=30)


# In[ ]:


train['Age'].hist(bins=30,color='darkred',alpha=0.7)


# In[ ]:


sns.countplot(x='SibSp',data=train)


# In[ ]:


train['Fare'].hist(color='green',bins=40,figsize=(8,4))


# **Data Cleaning**
# 

# In[ ]:


plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')


# In[ ]:


plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=test,palette='winter')


# In[ ]:


def impute_age_train(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age


# In[ ]:


def impute_age_test(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 42

        elif Pclass == 2:
            return 27

        else:
            return 24

    else:
        return Age


# In[ ]:


train['Age'] = train[['Age','Pclass']].apply(impute_age_train,axis=1)


# In[ ]:


test['Age'] = test[['Age','Pclass']].apply(impute_age_test,axis=1)


# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


train.drop('Cabin',axis=1,inplace=True)


# In[ ]:


test.drop('Cabin',axis=1,inplace=True)


# In[ ]:


print(train.dropna(inplace=True))


# In[ ]:


print(test.dropna(inplace=True))


# **Converting Categorical Features**

# In[ ]:


sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)


# In[ ]:


train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[ ]:


train = pd.concat([train,sex,embark],axis=1)


# In[ ]:


train.head()


# In[ ]:


sex = pd.get_dummies(test['Sex'],drop_first=True)
embark = pd.get_dummies(test['Embarked'],drop_first=True)


# In[ ]:


test.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[ ]:


test = pd.concat([test,sex,embark],axis=1)


# In[ ]:


test.head()


# **Building a Logistics Regression model**

# In[ ]:


X_train = train.drop('Survived',axis=1)


# In[ ]:


y_train = train['Survived']


# In[ ]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[ ]:


predictions = logmodel.predict(test)
print("Accuracy on training set: {}".format(logmodel.score(X_train,y_train)))


# In[ ]:


my_submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})


# In[ ]:


my_submission.to_csv('submission.csv', index=False)


# In[ ]:




