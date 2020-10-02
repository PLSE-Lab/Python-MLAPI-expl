#!/usr/bin/env python
# coding: utf-8

# In[16]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# First lets import the data and have a look

# In[17]:


# import data and view a summary
raw_data = pd.read_csv('../input/train.csv')
raw_data.head()


# In[18]:


print(raw_data.columns)
print(raw_data.isnull().sum())


# We're going to focus on 3 of the easier to manage columns that give the most information, Pclass, Sex and Embarked. If we group by these categories and take the mean of Survived then we get the percentage of people who survived in each category.

# In[19]:


raw_data.groupby(['Pclass', 'Sex', 'Embarked'])['Survived'].mean()


# This seems to give a lot of information, we ideally want to see values close to 0 or 1 as these tell us with confidence whether people in that category are likely to have survived or not. Almost all of the females in 1st class survived, whereas the majority of males in 2nd and 3rd class did not.

# In[20]:


# define independent and dependent variables and the features

y = raw_data.Survived
features = ['Pclass', 'Sex', 'Embarked']
X = raw_data[features].copy()


# Next we need to reformat some of the categorical data and allow for the 2 null values in 'Embarked'. I do this with a function which can then be reused on the test data

# In[21]:


#reformat data to change sex into binary
def reformat(x):
    d_sex = {'female': 0, 'male': 1}
    d_emb = {'C': 0, 'Q': 1, 'S': 2, '3':3}
    x['Sex'] = [d_sex[sex] for sex in x['Sex'].values]
    x['Embarked'] = x['Embarked'].fillna('3')
    x['Embarked'] = [d_emb[elem] for elem in x['Embarked'].values]
    
reformat(X)
X.head()


# Now scale the data so that features are given even weight

# In[22]:


scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# Then train the random forest generator model

# In[23]:


model = RandomForestClassifier()
model.fit(X_scaled,y)
cross_val_score(model , X_scaled , y , cv=5)


# Now I define a function to apply to the test data that will reformat, scale and run the model

# In[24]:


def applyModel(X_test):
    X = X_test[features].copy()
    reformat(X)
    X = scaler.transform(X)
    preds = model.predict(X)
    return preds


# In[25]:


X_test = pd.read_csv('../input/test.csv')
preds = applyModel(X_test)
output = pd.DataFrame({'PassengerId': X_test.PassengerId,
                       'Survived': preds})
output.to_csv('submission.csv', index=False)

