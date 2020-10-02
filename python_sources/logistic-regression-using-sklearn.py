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


train = pd.read_csv('../input/train.csv')
train.head()


# In[ ]:


test = pd.read_csv('../input/test.csv')
test.head()


# In[ ]:


submission = pd.read_csv('../input/gender_submission.csv')
submission.head()


# In[ ]:


train['Cabin'].isnull().values.ravel().sum()


# In[ ]:


len(train)


# In[ ]:


train['Pclass'].isnull().values.ravel().sum()


# In[ ]:


train['Sex'].isnull().values.ravel().sum()


# In[ ]:


train['Age'].isnull().values.ravel().sum()


# In[ ]:


train['SibSp'].isnull().values.ravel().sum()


# In[ ]:


train['Parch'].isnull().values.ravel().sum()


# In[ ]:


train['Fare'].isnull().values.ravel().sum()


# In[ ]:


train['Embarked'].isnull().values.ravel().sum()


# In[ ]:


del train['PassengerId']
del train['Cabin']
del train['Ticket']
del train['Name']


# In[ ]:


train.head()


# In[ ]:


train = train.dropna(subset=['Embarked']) 


# In[ ]:


len(train)


# In[ ]:


age = train['Age']


# In[ ]:


age.head()


# In[ ]:


del train['Age']


# In[ ]:


train.head()


# In[ ]:


train = pd.get_dummies(train)


# In[ ]:


train.head()


# In[ ]:


local_train = np.asarray(train)
age = np.asarray(age)
age = np.reshape(age, (889,1))
local_train.shape


# In[ ]:


data = np.concatenate((local_train, age), axis = 1)


# In[ ]:


data.shape


# In[ ]:


data[0][10]


# In[ ]:


data_train = []
for i in range(0,889):
    if np.isnan(data[i][10]):
        data_train.append(data[i])


# In[ ]:


len(data_train)


# In[ ]:


data_train = np.asarray(data_train)
data_train.shape


# In[ ]:


df = pd.DataFrame(data, index = None)


# In[ ]:


len(df)


# In[ ]:


df = df.dropna()


# In[ ]:


len(df)


# In[ ]:


data = np.asarray(df)


# In[ ]:


data.shape


# In[ ]:


x = data[:, 1:10]


# In[ ]:


x.shape


# In[ ]:


y = data[:, 10:]


# In[ ]:


y.shape


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


model1 = LinearRegression()
model1.fit(x, y)


# In[ ]:


x_test1 = data_train[:, 1:10]
x_test = data_train[:, :10]


# In[ ]:


y_test = model1.predict(x_test1)


# In[ ]:


y_test


# In[ ]:


y_test = np.round(y_test, decimals = 0)


# In[ ]:


y_test


# In[ ]:


for i in range(0,177):
    if y_test[i] <=0 :
        y_test[i] = abs(y_test[i])


# In[ ]:


y_test


# In[ ]:


data_new = np.concatenate((x_test, y_test), axis = 1)


# In[ ]:


data_new.shape


# In[ ]:


data = np.concatenate((data, data_new), axis = 0)


# In[ ]:


data.shape


# In[ ]:


y = data[:, 0:1]
x = data[:, 1:]


# In[ ]:


y


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42)


# In[ ]:


x_train.shape


# In[ ]:


y_train.shape


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


model = LogisticRegression()
model.fit(x_train,y_train)


# In[ ]:


y_predict = model.predict(x_test)


# In[ ]:


y_predict


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


print(accuracy_score(y_test, y_predict))


# In[ ]:


test.head()


# In[ ]:


del test['Cabin']
del test['PassengerId']
del test['Name']
del test['Ticket']


# In[ ]:


test.head()


# In[ ]:


test['Age'].isnull().values.ravel().sum()


# In[ ]:


test = pd.get_dummies(test)
age1 = test['Age']
age1 = np.asarray(age1)
del test['Age']
test1 = np.asarray(test)


# In[ ]:


age1 = np.reshape(age1, (418,1))
test1.shape


# In[ ]:


test1[0][1]


# In[ ]:


test1 = np.concatenate((test1, age1), axis = 1)


# In[ ]:


test1.shape


# In[ ]:


data_ = []
for i in range(0,418):
    if np.isnan(test1[i][9]):
        data_.append(test1[i])


# In[ ]:


len(data_)


# In[ ]:


data = np.asarray(data_)


# In[ ]:


data.shape


# In[ ]:




