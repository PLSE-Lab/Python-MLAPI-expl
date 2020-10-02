#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


def preprocess(data):
    data['Gender']=data['Sex'].apply(lambda a: 1 if a=='male' else 0)
    data=data.drop(['PassengerId','Name','Cabin','Embarked','Ticket','Sex'],axis=1)
    data=data.dropna()
    return data

datafile = '../input/train.csv'
data = pd.read_csv(datafile)
train=preprocess(data)

test = pd.read_csv('../input/test.csv')
passengerID=test['PassengerId']
test['Gender']=test['Sex'].apply(lambda a: 1 if a=='male' else 0)
test=test.drop(['PassengerId','Name','Cabin','Embarked','Ticket','Sex'],axis=1)


# In[6]:


test.isnull().sum()


# In[5]:


test['Age']=test['Age'].fillna(test['Age'].mean())
test['Fare']=test['Fare'].fillna(test['Fare'].mean())


# In[7]:


train.head()


# In[8]:


test.head()


# In[9]:


from sklearn.ensemble import RandomForestClassifier

X_train=train.drop(['Survived'],axis=1).values
y_train=train["Survived"].values
X_test=test.values


clf = RandomForestClassifier(n_estimators=1000,max_leaf_nodes=15)
clf=clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
predictions


# In[10]:


len(predictions)


# In[11]:


len(passengerID)


# In[12]:


result=pd.DataFrame({'PassengerID':passengerID,'Survived':predictions})
result.head()


# In[13]:


result.to_csv('submission.csv',index=False)


# In[16]:


print(os.listdir("../working"))

