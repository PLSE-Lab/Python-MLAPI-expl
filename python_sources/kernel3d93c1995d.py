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


train_df=pd.read_csv("/kaggle/input/titanic/train.csv")
train_df.head()


# In[ ]:


test_df=pd.read_csv("/kaggle/input/titanic/test.csv")
test_df.head()


# In[ ]:


combine=[train_df, test_df]
train_df


# In[ ]:


print(train_df.columns.values)


# In[ ]:


train_df.describe(percentiles=None, include=None, exclude=None)


# In[ ]:


train_df.describe(percentiles=None, include='O', exclude=None)


# In[ ]:


print('Percent of missing records in AGE is %.2f%%' %((train_df['Age'].isnull().sum()/train_df.shape[0])*100))


# In[ ]:


print('Percent of missing records in EMBARKED is %.2f%%' %((train_df['Embarked'].isnull().sum()/train_df.shape[0])*100))


# In[ ]:


print('Percent of missing records in CABIN is %.2f%%' %((train_df['Cabin'].isnull().sum()/train_df.shape[0])*100))


# In[ ]:


import math
age_mean=train_df["Age"].mean()
print(age_mean)


# In[ ]:


agetest_mean=test_df["Age"].mean()
print(agetest_mean)


# In[ ]:


num_rows=train_df.shape[0]
print(num_rows)


# In[ ]:


numtest_rows=test_df.shape[0]
print(numtest_rows)


# In[ ]:


for i in range(num_rows):
    if math.isnan(train_df.Age[i]):
        train_df.Age[i]=age_mean


# In[ ]:


for i in range(numtest_rows):
    if math.isnan(test_df.Age[i]):
        test_df.Age[i]=age_mean


# In[ ]:


train_df


# In[ ]:


test_df


# In[ ]:


train_df["Embarked"].fillna(train_df["Embarked"].value_counts().idxmax(), inplace=True)
train_df


# In[ ]:


test_df["Embarked"].fillna(test_df["Embarked"].value_counts().idxmax(), inplace=True)
test_df


# In[ ]:


train_df.drop(['Cabin'], axis=1, inplace=True)


# In[ ]:


test_df.drop(['Cabin'], axis=1, inplace=True)


# In[ ]:


test_df.isnull().sum()


# In[ ]:


train_df.isnull().sum()


# In[ ]:


train_df.head()


# In[ ]:


train_df = pd.get_dummies(train_df,columns=['Sex','Pclass'])
train_df.drop(['Sex_male'], axis=1, inplace=True)
train_df.drop(['Sex_female'], axis=1, inplace=True)
train_df


# In[ ]:


train_df.head()


# In[ ]:


train_df.isnull().sum()


# In[ ]:


test_df = pd.get_dummies(test_df,columns=['Sex','Pclass'])
test_df.drop(['Sex_male'], axis=1, inplace=True)
test_df.drop(['Sex_female'], axis=1, inplace=True)


# In[ ]:


test_df.head()


# In[ ]:


train_df = pd.get_dummies(train_df,columns=['Embarked'])
train_df.head()


# In[ ]:


test_df = pd.get_dummies(test_df,columns=['Embarked'])
test_df.head()


# In[ ]:


test_df.drop(['Name', 'Ticket'], axis=1, inplace=True)


# In[ ]:


train_df['TravelAlone']=np.where((train_df["SibSp"]+train_df["Parch"])>0, 0, 1)
train_df.drop(['SibSp', 'Parch', 'Name', 'Ticket'], axis=1, inplace=True)


# In[ ]:


train_df.head()


# In[ ]:


test_df['TravelAlone']=np.where((test_df["SibSp"]+test_df["Parch"])>0, 0, 1)
test_df.drop(['SibSp', 'Parch'], axis=1, inplace=True)


# In[ ]:


test_df.head()


# In[ ]:


train_df.isnull().sum()


# In[ ]:


test_df.isnull().sum()


# In[ ]:


print('Percent of missing records in FARE in TEST DATA is %.2f%%' %((test_df['Fare'].isnull().sum()/test_df.shape[0])*100))


# In[ ]:


test_df["Fare"].fillna(test_df["Fare"].value_counts().idxmax(), inplace=True)
test_df.head()


# In[ ]:


test_df.isnull().sum()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test  = train_test_split(train_df.drop('Survived',axis=1), train_df['Survived'], test_size=0.2, random_state=101)


# In[ ]:


from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()


# In[ ]:


log_model.fit(X_train, y_train)


# In[ ]:


predict = log_model.predict(X_test)
print(predict)


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test,predict))

