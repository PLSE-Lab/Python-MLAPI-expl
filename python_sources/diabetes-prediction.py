#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


pima = pd.read_csv("../input/diabetes.csv")
pima.head()


# In[ ]:


pima.isnull().sum()


# In[ ]:


pima.duplicated().sum()


# In[ ]:


X = pima.iloc[:,0:-1]


# In[ ]:


Y=pima.iloc[:,-1]


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)


# In[ ]:


clf = DecisionTreeClassifier()


# In[ ]:


clf = clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)


# In[ ]:


print("Accuracy: ",metrics.accuracy_score(Y_test, y_pred))


# In[ ]:




