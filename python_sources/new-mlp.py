#!/usr/bin/env python
# coding: utf-8

# In[23]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[3]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[4]:


train.head()


# In[7]:


train.columns


# In[53]:


feature = ["SibSp","Age","Pclass","Fare"]
label = "Survived"


# In[54]:


X_train = train[feature]
Y = train[label]


# In[55]:


enc = preprocessing.OrdinalEncoder()


# In[56]:


enc.fit(train['Sex'].values.reshape(-1,1))


# In[57]:


enc.categories


# In[58]:


X_train["Male"] = 0
X_train["Female"] = 1


# In[59]:


X_train.isnull().sum()


# In[60]:


X_train["Age"] = X_train["Age"].fillna(0)


# In[61]:


MLPA = MLPClassifier(hidden_layer_sizes=(10,),max_iter=20)


# In[62]:


MLPA.fit(X_train,Y)


# In[63]:


MLPA.score(X_train,Y)


# In[64]:


X_test = test[feature]


# In[65]:


enc.fit(test["Sex"].values.reshape(-1,1))


# In[66]:


X_test["Male"] = 0
X_test["Female"] = 1


# In[67]:


X_test.isnull().sum()


# In[68]:


X_test["Fare"].fillna(0)


# In[69]:


X_test.isnull().sum()


# In[71]:


X_test["Fare"] = X_test["Fare"].fillna(0)


# In[72]:


X_test["Age"] = X_test["Age"].fillna(0)


# In[75]:


X_test.isnull().sum()


# In[76]:


MLPA.predict(X_test)


# In[77]:


test["Survived"] = MLPA.predict(X_test)


# In[78]:


test[["PassengerId","Survived"]].to_csv("NEW MLP",index = False)


# In[ ]:




