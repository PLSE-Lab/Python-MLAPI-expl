#!/usr/bin/env python
# coding: utf-8

# In[282]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[283]:


train_data= pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
gender_submisson = pd.read_csv('../input/gender_submission.csv')


# In[284]:


train_data.head()


# In[285]:


test_data.info()


# In[286]:


train_data.info()


# There are null values in Age,cabin,Embarked. Lets replace null values in Age with mean, Embarked with Mean and remove cabin

# In[287]:


# Update sex column to numerical

train_data['Sex'] = train_data['Sex'].apply(lambda x: 1 if x=='male' else 0)
train_data.head()


# In[288]:


test_data['Sex'] = test_data['Sex'].apply(lambda x: 1 if x=='male' else 0)


# In[289]:


#print(train_data['Age'].values)
mean_age = train_data.Age.mean()
train_data['Age'].fillna(value = mean_age, inplace = True)


# In[290]:


#print(train_data['Age'].values)
mean_age = test_data.Age.mean()
test_data['Age'].fillna(value = mean_age, inplace = True)


# In[ ]:





# In[291]:


print(train_data['Age'].values)


# In[292]:


train_data.isnull().sum()


# In[293]:


# Select the desired features
train_data = train_data[[ 'PassengerId', 'Survived', 'Pclass','Name','Sex', 'Age', 'SibSp', 'Parch', 'Ticket','Fare'] ]
# Select the desired features
test_data = test_data[[ 'PassengerId', 'Pclass','Name','Sex', 'Age', 'SibSp', 'Parch', 'Ticket','Fare'] ]


# In[294]:


train_data.head()


# In[295]:


# Create a Upper column
train_data['Upper'] = train_data['Pclass'].apply(lambda x: 1 if x==1 else 0)

# Create a Middle column
train_data['Middle'] = train_data['Pclass'].apply(lambda x: 1 if x==2 else 0)

# Create a Lower column
train_data['Lower'] = train_data['Pclass'].apply(lambda x: 1 if x==3 else 0)


# In[296]:


# Create a Upper column
test_data['Upper'] = test_data['Pclass'].apply(lambda x: 1 if x==1 else 0)

# Create a Middle column
test_data['Middle'] = test_data['Pclass'].apply(lambda x: 1 if x==2 else 0)

# Create a Lower column
test_data['Lower'] = test_data['Pclass'].apply(lambda x: 1 if x==3 else 0)


# In[297]:


train_data.isnull().sum()


# In[298]:


# Select the desired features
selected_features = ['Sex', 'Age', 'Upper', 'Middle', 'Lower', 'SibSp', 'Parch', 'Fare'] #

# Select the desired features
features = train_data[selected_features]
survival = train_data['Survived']


# In[299]:


features.head()


# In[300]:


# Perform train, test, split

train_features, test_features, train_labels, test_labels = train_test_split(features, survival, train_size = 0.8)


# In[301]:


test_data1 = test_data[selected_features]


# In[302]:


test_data1.isnull().sum()


# In[303]:


#print(train_data['Age'].values)
mean_fare = test_data1.Fare.mean()
test_data1['Fare'].fillna(value = mean_fare, inplace = True)


# In[304]:


# Scale the feature data so it has mean = 0 and standard deviation = 1
normalize = StandardScaler()
train_features = normalize.fit_transform(train_features)
test_features = normalize.transform(test_features)
test_data1 =normalize.transform(test_data1) 


# In[305]:


# Create and train the model
model = LogisticRegression()
model.fit(train_features, train_labels)


# In[306]:


# Score the model on the train data
print(model.score(train_features, train_labels))


# In[307]:


# Score the model on the test data
print(model.score(test_features, test_labels))


# In[308]:


print(list(zip(selected_features, model.coef_[0])))


# In[309]:


prdctions=model.predict_proba(test_data1)[:,1]
print(prdctions)


# In[310]:


print(gender_submisson.PassengerId)


# In[311]:


my_submission = pd.DataFrame({'PassengerId': gender_submisson.PassengerId, 'Survived': prdctions})


# In[312]:


my_submission = [('PassengerId', gender_submisson.PassengerId),('Survived', model.predict_proba(test_features))      ]

my_submission = pd.DataFrame(my_submission)
my_submission.to_csv('my_submission.csv')

