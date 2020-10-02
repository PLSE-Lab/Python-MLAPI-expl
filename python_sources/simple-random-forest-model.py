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


train = pd.read_csv('/kaggle/input/learn-together/train.csv')


# In[ ]:


test = pd.read_csv('/kaggle/input/learn-together/test.csv')


# In[ ]:


train.head()


# In[ ]:


test.head()


# ## Train Test Split

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X = train.drop(['Id', 'Cover_Type'], axis=1)
y = train['Cover_Type']


# Let's use 70% of the Data for training, and 30% for validation.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# ## Random Forests

# In[ ]:


from sklearn.ensemble import RandomForestClassifier 


# We'll then train a simple Random Forest Classifier with 100 trees.

# In[ ]:


rfc = RandomForestClassifier(n_estimators=100)


# In[ ]:


rfc.fit(X_train, y_train)


# ## Prediction and Evaluation

# In[ ]:


from sklearn.metrics import classification_report, accuracy_score


# In[ ]:


rfc.score(X_train, y_train)


# In[ ]:


predictions = rfc.predict(X_test)


# In[ ]:


accuracy_score(y_test, predictions)


# In[ ]:


print(classification_report(y_test, predictions))


# ## Finding the 'Cover_Type' for Test

# In[ ]:


test_Id = test['Id'] #store tests' Id column for the output file


# In[ ]:


test = test.drop('Id', axis=1) #delete the Id column for the prediction


# In[ ]:


test.head()


# In[ ]:


test_pred = rfc.predict(test)


# Save test predictions to file

# In[ ]:


output = pd.DataFrame({'Id': test_Id,
                       'Cover_Type': test_pred})
output.to_csv('submission.csv', index=False)


# In[ ]:




