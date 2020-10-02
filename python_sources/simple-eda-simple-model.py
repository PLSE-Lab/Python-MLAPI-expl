#!/usr/bin/env python
# coding: utf-8

# Version: 01
# Last updated: 25.03.2020

# # Main imports

# In[ ]:


import numpy as np 
import pandas as pd 
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Load data

# In[ ]:


temperature_submission = pd.read_csv('/kaggle/input/killer-shrimp-invasion/temperature_submission.csv')
test = pd.read_csv('/kaggle/input/killer-shrimp-invasion/test.csv')
train = pd.read_csv('/kaggle/input/killer-shrimp-invasion/train.csv')


# Let's take a look at the data

# In[ ]:


temperature_submission.head(5)


# In[ ]:


test.head(5)


# In[ ]:


train.head(5)


# # EDA

# In[ ]:


import seaborn as sns


# In[ ]:


ax = sns.countplot(train['Presence'])


# We can see there is an imbalance for class spreading in train data

# In[ ]:


train.corr()


# In[ ]:


#ax = sns.lineplot(train['Salinity_today'], train['Presence'])


# In[ ]:


#ax = sns.lineplot(train['Temperature_today'], train['Presence'])


# In[ ]:


#ax = sns.lineplot(train['Substrate'], train['Presence'])


# In[ ]:


#ax = sns.lineplot(train['Depth'], train['Presence'])


# In[ ]:


#ax = sns.lineplot(train['Exposure'], train['Presence'])


# # Data Cleaning

# In[ ]:


print('Count of missing values in train data = ', train.isnull().sum(axis=1).sum())


# In[ ]:


print('Count of missing values in test data = ', test.isnull().sum(axis=1).sum())


# In[ ]:


train = train.fillna(method='ffill')
test = test.fillna(method='ffill')


# In[ ]:


print('Count of missing values in train data = ', train.isnull().sum(axis=1).sum())


# In[ ]:


print('Count of missing values in test data = ', test.isnull().sum(axis=1).sum())


# # FE

# todo

# # MODELLING

# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier
clf = ExtraTreesClassifier(n_estimators=100, random_state=0).fit(train.drop(['pointid', 'Presence'], axis=1), train['Presence'])


# In[ ]:


predictions = clf.predict(test.drop(['pointid'], axis=1))


# In[ ]:


temperature_submission['Presence'] = predictions
temperature_submission.head(5)


# Create submission

# In[ ]:


temperature_submission.to_csv('temperature_submission.csv', index=False)

