#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df_products = pd.read_csv("../input/train.csv")


# In[ ]:


df_products.head()


# In[ ]:


df_y = df_products["target"]


# In[ ]:


df_y.head()


# In[ ]:


df_X = df_products.drop('id', axis=1).drop('target', axis=1)


# In[ ]:


df_X.head()


# In[ ]:


df_X.to_csv("train_X.csv", header=True, index=False)
df_y.to_csv("train_y.csv", header=True, index=False)


# In[ ]:


pd.read_csv("train_X.csv")


# In[ ]:


X = df_X.values


# In[ ]:


X


# In[ ]:


y = df_y.values


# In[ ]:


y = y.reshape(-1)


# In[ ]:


print("y shape:", y.shape)


# In[ ]:


X.shape


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


print("X train:", X_train.shape)
print("y train:", y_train.shape)
print()
print("X test: ", X_test.shape)
print("y test: ", y_test.shape)


# In[ ]:


model = LogisticRegression(random_state=42)


# In[ ]:


model.fit(X_train, y_train)


# In[ ]:


LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1, penalty='l2', random_state=45, solver='liblinear', tol=0.0001, verbose=0, warm_start=False)


# In[ ]:


y_train_pred = model.predict(X_train)


# In[ ]:


y_test_pred = model.predict(X_test)


# In[ ]:


accuracy_train = accuracy_score(y_train, y_train_pred)
accuracy_test = accuracy_score(y_test, y_test_pred)


# In[ ]:


print("Training accuracy: {0:.3f}%".format(accuracy_train * 100))
print("Test accuracy: {0:.3f}%".format(accuracy_test * 100))

