#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


fake_data = pd.read_csv('../input/fake-and-real-news-dataset/Fake.csv')
real_data = pd.read_csv('../input/fake-and-real-news-dataset/True.csv')


# In[ ]:


print(fake_data.isnull().sum())
print(real_data.isnull().sum())


# In[ ]:


print(fake_data.shape)
print(real_data.shape)


# In[ ]:


fake_data['subject'].value_counts()


# In[ ]:


real_data['subject'].value_counts()


# In[ ]:


fake_data['Real'] = False
real_data['Real'] = True


# In[ ]:


df = pd.concat([fake_data,real_data])
df.head()


# I choose subject feature to look up, because in true data we have only politicsNews and worldNews subjects.

# In[ ]:


df = df[['subject','Real']]
df.isnull().sum()


# In[ ]:


one_hot_encoded_training_predictors = pd.get_dummies(df)


# In[ ]:


X = one_hot_encoded_training_predictors.iloc[:,0:4]
y = one_hot_encoded_training_predictors.iloc[:,4]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1)


# In[ ]:


dt = DecisionTreeClassifier(random_state=1)
dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)
print(accuracy_score(y_test,y_pred))

