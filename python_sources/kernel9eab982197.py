#!/usr/bin/env python
# coding: utf-8

# In[5]:


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


# In[6]:


dataset = pd.read_csv("../input/heart.csv")


# In[7]:


dataset.describe()


# dataset.isna().any()

# In[8]:


dataset.head()


# 

# In[10]:


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 13].values


# In[11]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X = sc.fit_transform(X)


# 

# In[12]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.75, random_state = 1)


# In[13]:


from sklearn.svm import SVC
svc = SVC()


# In[14]:


svc.fit(X_train, y_train)


# In[16]:


y_pred = svc.predict(X_test)


# In[18]:


from sklearn.metrics import confusion_matrix, accuracy_score


# In[19]:


cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)


# In[20]:


cm

