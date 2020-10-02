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


# This notebook is an introduction to Kaggle based on live sessions from the ConnectAI meetup.

# # Loading the data

# In[ ]:


# first we load in the train dataset and take a peek at it...
train = pd.read_csv('/kaggle/input/titanic/train.csv')
train.head()


# In[ ]:


# ...and do the same for the test data...
test = pd.read_csv('/kaggle/input/titanic/test.csv')
test.head()


# Notice that train contains a 'Survived' column and test does not. Our job is to produce the 'Survived' column for the test data, using patterns we learn from the train data. In other words to predict which passengers survive.

# # Data Exploration
# TODO

# # Data Preprocessing
# Before we can feed our data into a machine learning model, we need to do a little cleanup. In particular, there are two main issues we need to address.
# 1. Non-numeric data: Most machine learning models deal only with numbers. Since our data includes words, we'll either have to drop those, or convert them to numbers somehow.
# 2. Missing values: NaN stands for "not a number." This is pandas-speak for missing data. Most models don't know how to deal with missing data, so we'll need to fill those in the missing values.

# We'll start by dropping some problematic columns. PassengerId is just an index that doesn't contain any useful information. Name, Ticket, and Cabin are all word columns that can't be converted to useful numbers in an obvious way. As a quick and dirty start, we'll just chuck them out. If you want to improve this notebook, you might consider how you could extract useful information from those columns.
# 

# In[ ]:


train = train.drop(['PassengerId','Name','Ticket', 'Cabin'], axis=1)
train.head()


# The 'Sex' column is also words, but given that it only has two values, and seems likely to be quite important for our predictions ("Women and children first!") let's go ahead and turn it into numbers. We'll just code 'male' as 1 and 'female' as 0.

# In[ ]:


train['Sex'] = train['Sex'].replace({'male':1, 'female':0})
train.head()


# In[ ]:


train['Embarked'].unique()


# 'Embarked' has three different values, representing the port the passenger departed from. The most obvious way to encode this, similar to what we did with 'Sex', would be to encode those values as 0, 1, and 2, but this tells our model that (for instance) C is "halfway between" S and Q, which doesn't exactly make sense. It might be better to use "one-hot encoding", which means we create a column for each value, and set it to one if that was the value, or zero otherwise. Fortunately, pandas does this for us.

# In[ ]:


train = pd.get_dummies(train)
train.head()


# Okay, we've dealt with all the words. Now we just have to fill in the missing values. Let's see how many we've got.

# In[ ]:


train.isna().sum()


# In the train dataframe, only the 'Age' column has missing values. There are many ways of dealing with missing values. For now, we'll replace them with the median value.

# In[ ]:


median_age = train['Age'].median()
median_age


# In[ ]:


train['Age'] = train['Age'].fillna(median_age)


# # Modeling
# Phew! Our data is finally in a format that can be fed into a machine learning model. This could be any algorithm that takes as input the feature columns and outputs a prediction for 'Survived.' Common model types include logistic regression, random forest, or neural network. Let's try a neural network. We'll use the implementation from sk-learn, a popular machine learning library.

# In[ ]:


y = train['Survived']
X = train.drop('Survived', axis=1)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


model = RandomForestClassifier()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(X, y)


# In[ ]:


model.fit(X_train, y_train);


# In[ ]:


preds = model.predict(X_valid)
preds[:10]


# In[ ]:


def accuracy(preds, target): return (preds==target).sum()/len(preds)
accuracy(preds, y_valid)


# In[ ]:


test = pd.read_csv('/kaggle/input/titanic/test.csv')
PassengerId = test['PassengerId'] # we save the passenger ids because we'll need them for our submission
test = test.drop(['PassengerId','Name','Ticket', 'Cabin'], axis=1)
test['Sex'] = test['Sex'].replace({'male':1, 'female':0})
test['Age'] = test['Age'].fillna(median_age)
test['Fare'] = test['Fare'].fillna(train['Fare'].mean()) # it turned out that test had a missing value in Fare that needed to be filled in
test = pd.get_dummies(test)
test.head()


# In[ ]:


preds = model.predict(test)
sub = pd.DataFrame({'PassengerId': PassengerId, 'Survived': preds})
sub.head()


# In[ ]:


sub.to_csv('survived.csv', index=False)


# We're finally ready to submit our predictions and see how we did!
# 1. Click the "Commit" button in the upper right.
# 2. From the window that pops up, click "Open Version."
# 3. Select "Output" from the list on the left.
# 4. Click "Submit to Competition"

# In[ ]:




