#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
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
test = pd.read_csv("../input/titanic-machine-learning-from-disaster/test.csv")
train = pd.read_csv("../input/titanic-machine-learning-from-disaster/train.csv")


# In[ ]:


test.head(5)


# In[ ]:


train.head(5)


# In[ ]:


test.shape


# In[ ]:


train.shape
len(train)


# In[ ]:


train_required = train[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Survived','PassengerId']]


# In[ ]:


train_required.head(5)


# In[ ]:


len(train_required)


# In[ ]:


train_required.isnull()


# In[ ]:


train_required['Embarked'].isnull().values.any()


# In[ ]:


train_required['Age'].fillna(train_required['Age'].mean(),inplace=True)


# In[ ]:


train_required['Age'].mean()


# In[ ]:


train_required = train_required.drop('Cabin',axis=1)


# In[ ]:


train_required.head()


# In[ ]:


train_required = train_required.dropna()


# In[ ]:


len(train_required)


# In[ ]:


from sklearn import preprocessing
lblEncoder = preprocessing.LabelEncoder()


# In[ ]:


train_required['Sex'] = lblEncoder.fit_transform(train_required['Sex'])
train_required['Embarked'] = lblEncoder.fit_transform(train_required['Embarked'])
train_required.head()


# In[ ]:


from sklearn.naive_bayes import GaussianNB
ngnb =GaussianNB()


# In[ ]:


X_train = train_required.drop('Survived',axis=1)
Y_train = train_required['Survived']

X_train.head(5)


# In[ ]:


Y_train.head(5)


# In[ ]:


ngnb.fit(X_train,Y_train)


# In[ ]:



test = pd.read_csv("../input/titanic-machine-learning-from-disaster/test.csv")
test.head(5)


# In[ ]:


test = test[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','PassengerId']]
test.head(5)


# In[ ]:


test['Embarked'].isnull().values.any()


# In[ ]:


test['Age'].fillna(test['Age'].mean(),inplace=True)


# In[ ]:


len(test)


# In[ ]:


test_req = test.dropna()
len(test_req)


# In[ ]:


from sklearn import preprocessing
lblEncoderTest = preprocessing.LabelEncoder()
test_req['Sex'] = lblEncoderTest.fit_transform(test_req['Sex'])
test_req['Embarked'] = lblEncoderTest.fit_transform(test_req['Embarked'])
test_req.head()


# In[ ]:


y_pred = ngnb.predict(test_req)


# In[ ]:


from sklearn import metrics


# In[ ]:


metrics.accuracy_score(y_test, y_pred)


# In[ ]:


submission = pd.DataFrame({"PassengerId": test_req["PassengerId"], "Survived": y_pred})

submission.to_csv("Titanic Predictions 1.csv", index=False)

submission.head()


# In[ ]:




