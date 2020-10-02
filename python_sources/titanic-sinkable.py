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


#Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


#Reading the train and test data
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# In[ ]:


#Checking data
train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


#Check info
train_df.info()
print('-----------------------------------------------------------------')
test_df.info()


# Except the target value, we have 2 float, 4 integer and 5 object dtypes columns.
# Now we will explore our data.

# In[ ]:


#Remove the target feature i.e. Survived
y1 = train_df['Survived'].copy()
train_df.drop(columns=['Survived'],inplace=True)


# In[ ]:


train_df.describe()


# In[ ]:


train_df.isnull().sum()


# In[ ]:


test_df.isnull().sum()


# In[ ]:


# We have missing values in Age, Cabin and Embarked
# Cabin has a lot of missing values so we will drop the column
train_df.drop(columns=['Cabin'],inplace=True)
test_df.drop(columns=['Cabin'],inplace=True)
# Age is a numerical data, so we will use median to fill the missing values
train_df['Age'].fillna(train_df['Age'].median(),inplace=True)
test_df['Age'].fillna(test_df['Age'].median(),inplace=True)

# Fare in test data
test_df['Fare'].fillna(test_df['Fare'].median(),inplace=True)


# In[ ]:


# Embarked in train data
# Find category and it's counts
train_df['Embarked'].value_counts().plot(kind='bar',figsize=(6,4), title='Embarked')


# In[ ]:


# We will replace S which has the highest frequency in the place of missing values
train_df['Embarked'].fillna('S',inplace=True)


# We will drop unnecessary columns from our data that do not have any significant impact

# In[ ]:


# PassengerId, Ticket are not of any use
train_df.drop(columns=['PassengerId','Ticket'],inplace=True)
test_df.drop(columns=['PassengerId','Ticket'],inplace=True)


# In[ ]:


# Feature Enginnering
train_df['Title'] = train_df['Name'].apply(lambda x: x[x.find(',')+2:x.find('.')])
test_df['Title'] = test_df['Name'].apply(lambda x: x[x.find(',')+2:x.find('.')])


# In[ ]:


train_df['Title'] = train_df['Title'].replace(['Mme','Ms'],'Mrs')
train_df['Title'] = train_df['Title'].replace(['Mlle','Lady'],'Miss')
train_df['Title'] = train_df['Title'].replace(['the Countess',
                                               'Capt', 'Col','Don', 
                                               'Dr', 'Major', 'Rev', 
                                               'Sir', 'Jonkheer', 'Dona'], 'Others')

test_df['Title'] = test_df['Title'].replace(['Mme','Ms'],'Mrs')
test_df['Title'] = test_df['Title'].replace(['Mlle','Lady'],'Miss')
test_df['Title'] = test_df['Title'].replace(['the Countess',
                                               'Capt', 'Col','Don', 
                                               'Dr', 'Major', 'Rev', 
                                               'Sir', 'Jonkheer', 'Dona'], 'Others')


# In[ ]:


train_df['Title'].value_counts().plot(kind='bar', figsize=(6,4), title='Title')


# In[ ]:


test_df['Title'].value_counts().plot(kind='bar', figsize=(6,4), title='Title')


# In[ ]:


# Drop Name Column
train_df.drop(columns=['Name'],inplace=True)
test_df.drop(columns=['Name'],inplace=True)


# In[ ]:


train_df = pd.get_dummies(columns=['Title','Embarked','Sex'],data=train_df)
test_df = pd.get_dummies(columns=['Title','Embarked','Sex'],data=test_df)


# In[ ]:


#Training data
x = train_df.iloc[:,:].values
y = y1.values

#Test data
test = test_df.iloc[:,:].values


# In[ ]:


from sklearn.model_selection import train_test_split
xtr,xvl,ytr,yvl = train_test_split(x,y,test_size=0.25,random_state=0)


# <h1>Modelling</h1>

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
rf.fit(xtr,ytr)
rf.score(xtr,ytr)


# In[ ]:


y_pred = rf.predict(xvl)


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred,yvl)
cm


# In[ ]:


y_pred2 = rf.predict(test)


# In[ ]:


submission = pd.read_csv('../input/gender_submission.csv')
df = pd.DataFrame({'Survived':y_pred2})
submission.update(df)


# In[ ]:


from lightgbm import LGBMClassifier


# In[ ]:


lgb = LGBMClassifier(objective='binary',random_state=0)
lgb.fit(x,y)
lgb.score(x,y)


# In[ ]:


y_pred2 = lgb.predict(test)
submission = pd.read_csv('../input/gender_submission.csv')
df = pd.DataFrame({'Survived':y_pred2})
submission.update(df)


# In[ ]:




