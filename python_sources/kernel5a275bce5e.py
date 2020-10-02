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


df = pd.read_csv('../input/train.csv')
df.describe(include='all')


# In[ ]:


df.drop(columns=['PassengerId', 'Name', 'Cabin', 'Ticket'], inplace=True)


# In[ ]:


df.head(5)


# In[ ]:


df_dummy = pd.get_dummies(df)
df_dummy.head(4)


# In[ ]:


df_dummy.describe(include='all')


# In[ ]:


df_dummy['Age'].fillna(df_dummy['Age'].median(), inplace=True)
df_dummy.describe(include='all')


# In[ ]:


df_dummy.head(3)


# In[ ]:


from sklearn.model_selection import train_test_split
X = df_dummy.drop(columns=['Survived'])
y = df_dummy['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier as randomforest
rfr = randomforest(random_state=0, n_estimators=100)

rfr.fit(X_train, y_train)


# In[ ]:


rfr.score(X_test, y_test)


# # training to all data in train.csv

# In[ ]:


rfr.fit(X, y)


# In[ ]:


testdata = pd.read_csv('../input/test.csv', index_col=0)
testdata.drop(columns=['Name', 'Cabin', 'Ticket'], inplace=True)
testdata_dummy = pd.get_dummies(testdata)
testdata_dummy.describe(include='all')


# In[ ]:


testdata_dummy['Age'].fillna(testdata_dummy['Age'].median(), inplace=True)
testdata_dummy['Fare'].fillna(testdata_dummy['Fare'].median(), inplace=True)


# In[ ]:


testdata_dummy.describe(include='all')


# In[ ]:


gender_submission = pd.read_csv('../input/gender_submission.csv', index_col=0)
test = pd.concat([testdata_dummy, gender_submission], axis=1)
test.head(5)


# In[ ]:


test.describe(include='all')


# In[ ]:


X_test = test.drop(columns=['Survived'])
y_test = test['Survived']


# In[ ]:


rfr.score(X_test, y_test)


# In[ ]:


y_pred = rfr.predict(X_test)
results = pd.DataFrame(X_test)
results['Survived'] = y_pred
results


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# In[ ]:


results.to_csv('Predict.csv', columns=['Survived'])


# In[ ]:




