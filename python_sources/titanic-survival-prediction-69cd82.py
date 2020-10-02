#!/usr/bin/env python
# coding: utf-8

# # Predicting Titanic Survival

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data_train = pd.read_csv('../input/train.csv')
data_test = pd.read_csv('../input/test.csv')
data_train.head()


# ## Exploratory Data Analysis

# In[ ]:


data_train.describe().T


# In[ ]:


data_train.info()


# We see that there are some missing value in 'Age' column, we will deal with it later. For now we will continue on our exploration of datasets.

# In[ ]:


sns.set_style('darkgrid')
sns.countplot(data=data_train, x='Survived', hue='Sex')


# We can see that there is more female that survive than male.

# In[ ]:


sns.countplot(data=data_train, x='Survived',hue='Pclass')


# Based off passenger class there is more likely the passenger with higher class tend to survive.

# In[ ]:


sns.distplot(data_train['Age'].dropna(), bins=30)


# ## Data Processing 

# In[ ]:


figure = plt.figure(figsize=(10,6))
sns.boxplot(data=data_train, x='Pclass', y='Age')


# We want to fill the missing value in Age. Looking at the boxplot above we can say that older person is more likely to afford higher class ticket. So based off this interpretation we will fill the missing value with age's average based on passenger class.

# In[ ]:


first_mean = round(data_train[data_train['Pclass'] == 1]['Age'].dropna().mean())
second_mean = round(data_train[data_train['Pclass'] == 2]['Age'].dropna().mean())
third_mean = round(data_train[data_train['Pclass'] == 3]['Age'].dropna().mean())

# creating function to fill missing age
def filling(col):
    Age = col[0]
    Pclass = col[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return first_mean
        elif Pclass == 2:
            return second_mean
        else:
            return third_mean
    else:
        return Age

data_train['Age'] = data_train[['Age', 'Pclass']].apply(filling, axis=1)


# Since Sex and Embarked columns are categorical data, we will use dummy variables to get their values for our machine learning model.

# In[ ]:


sex = pd.get_dummies(data_train['Sex'],drop_first=True)
embarked = pd.get_dummies(data_train['Embarked'],drop_first=True)


# Finally we've got our X_train and y_train

# In[ ]:


X_train = pd.concat([data_train[['Age', 'SibSp', 'Parch']], sex, embarked], axis=1)
y_train = data_train['Survived']


# ### Next we will repeat the same process as above on our test data so we can use it for our prediction later.

# In[ ]:


data_test['Age'] = data_test[['Age', 'Pclass']].apply(filling, axis=1)
sex = pd.get_dummies(data_test['Sex'],drop_first=True)
embarked = pd.get_dummies(data_test['Embarked'],drop_first=True)
X_test = pd.concat([data_test[['Age', 'SibSp', 'Parch']], sex, embarked], axis=1)


# ## Machine Learning Using Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
model.score(X_train, y_train)


# In[ ]:


predictions = model.predict(X_test)
data_test['Survived'] = predictions


# In[ ]:


submit = data_test[['PassengerId', 'Survived']]
submit.to_csv('submission.csv', index=False)

